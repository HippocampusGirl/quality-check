import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path
from time import time_ns
from typing import Any, Callable, Iterator, Type

import bitsandbytes
import torch
from accelerate.accelerator import Accelerator
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import (
    DDPCommunicationHookType,
    DistributedDataParallelKwargs,
    FullyShardedDataParallelPlugin,
    ProfileKwargs,
    TorchDynamoPlugin,
)
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.datamodule import DataModule
from ..data.schema import Datastore
from ..logging import logger
from ..utils import Timer, TrainingState
from .base import Model
from .discriminator import Discriminator, gradient_penalty
from .loss import LPIPS, AutoencoderLoss
from .model import (
    get_discriminator_model,
    get_learning_rate_scheduler,
    get_model,
    image_size,
    model_class,
    model_name,
)
from .validate import validate

torch._dynamo.config.cache_size_limit = 1 << 10
torch.set_float32_matmul_precision("high")


def epoch(  # noqa: C901
    state: TrainingState,
    accelerator: Accelerator,
    autoencoder: Model,
    autoencoder_loss: AutoencoderLoss,
    discriminator: Model,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    checkpoint: Callable[[], None],
    checkpoint_steps: int,
    validate: Callable[[], None],
    val_steps: int,
    max_step_count: int | None = None,
) -> None:
    autoencoder.model = autoencoder.model.train()
    discriminator.model = discriminator.model.train()

    data_timer = Timer()
    model_timer = Timer()

    data_timer.start()

    reconstructed_images: torch.Tensor | None = None
    average_autoencoder_loss: torch.Tensor | None = None
    average_discriminator_loss: torch.Tensor | None = None
    with (
        tqdm(
            total=len(train_dataloader),
            leave=False,
            disable=not accelerator.is_main_process,
            unit="steps" + " ",
            position=0,
        ) as progress_bar,
        accelerator.profile() as profile,
    ):
        progress_bar.set_description(f"Epoch {state.epoch_index.item() + 1}")
        for images, _ in islice(train_dataloader, max_step_count):
            batch_size = images.size(0)
            images = VaeImageProcessor.normalize(images)

            data_timer.stop()
            model_timer.start()

            step = state.step_index // accelerator.gradient_accumulation_steps
            is_autoencoder_step = (step % 2) == 0

            if is_autoencoder_step or reconstructed_images is None:
                with accelerator.accumulate(autoencoder.model):
                    (reconstructed_images, commit_loss) = autoencoder.model(
                        images, return_dict=False
                    )
                    if reconstructed_images is None:
                        raise ValueError

                    loss = autoencoder_loss(images, reconstructed_images)
                    loss += commit_loss

                    average_autoencoder_loss = accelerator.gather(
                        loss.repeat(batch_size)
                    ).mean()

                    # with torch._dynamo.compiled_autograd.enable(
                    #     torch.compile(**accelerator.state.dynamo_plugin.to_kwargs())
                    # ):
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(autoencoder.model.parameters(), 1.0)
                    autoencoder.optimizer.step()
                    autoencoder.learning_rate_scheduler.step()
                autoencoder.optimizer.zero_grad(set_to_none=True)
            else:
                with accelerator.accumulate(discriminator.model):
                    reconstructed_images = reconstructed_images.detach_()
                    reconstructed_score = discriminator.model(reconstructed_images)

                    images = images.requires_grad_()
                    score = discriminator.model(images)

                    loss = (
                        torch.nn.functional.relu(1.0 + reconstructed_score)
                        + torch.nn.functional.relu(1.0 - score)
                    ).mean()
                    loss += gradient_penalty(score, images)

                    average_discriminator_loss = accelerator.gather(
                        loss.repeat(batch_size)
                    ).mean()

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(discriminator.model.parameters(), 1.0)
                    discriminator.optimizer.step()
                    discriminator.learning_rate_scheduler.step()
                discriminator.optimizer.zero_grad(set_to_none=True)

            model_timer.stop()

            progress_bar.update(1)
            if state.step_index != 0:
                if accelerator.sync_gradients and accelerator.is_main_process:
                    learning_rate = autoencoder.learning_rate_scheduler.get_last_lr()[0]
                    logs = dict(
                        learning_rate=learning_rate,
                        epoch=state.epoch_index.item(),
                        data_time=data_timer.value,
                        model_time=model_timer.value,
                    )
                    progress_bar.set_postfix(**logs)
                    if average_autoencoder_loss is not None:
                        logs["autoencoder_train_loss"] = average_autoencoder_loss.item()
                        average_autoencoder_loss = None
                    if average_discriminator_loss is not None:
                        logs["discriminator_train_loss"] = (
                            average_discriminator_loss.item()
                        )
                        average_discriminator_loss = None
                    accelerator.log(logs, step=state.step_index.item())

                    if state.step_index % checkpoint_steps == 0:
                        checkpoint()
                if state.step_index % val_steps == 0:
                    validate()
            state.step_index += 1

            if profile is not None:
                profile.step()

            data_timer.start()


@dataclass
class Trainer:
    datastore: Datastore
    data_module_class: Type[DataModule]
    seed: int

    batch_size: int
    gradient_accumulation_steps: int

    epoch_count: int
    max_step_count: int | None = None

    is_profile: bool = False

    val_steps: int = 100
    val_count: int = 50
    checkpoint_steps: int = 500
    # https://github.com/CompVis/taming-transformers/issues/93
    discriminator_warmup_steps: int = 10000

    artifact_path: Path = field(init=False)
    accelerator: Accelerator = field(init=False)
    checkpoint_prefix: str = field(init=False)

    def __post_init__(self) -> None:
        if self.datastore.cache_path is None:
            raise ValueError("Datastore cache path is None")
        prefix = f"data-{self.data_module_class.name}_model-{model_name}"
        self.checkpoint_prefix = f"{prefix}_step-"
        self.artifact_path = self.datastore.cache_path

        fullgraph: bool = True
        mixed_precision = "bf16"
        kwargs_handlers: list[Any] = list()
        # mixed_precision = "fp8"
        # if mixed_precision == "fp8":
        #     fullgraph = False
        #     fp8_recipe_kwargs = FP8RecipeKwargs(backend="te", fp8_format="HYBRID")
        #     kwargs_handlers.append(fp8_recipe_kwargs)
        fsdp_plugin: FullyShardedDataParallelPlugin | None = None
        # deepspeed_plugin: DeepSpeedPlugin

        if self.is_profile:
            output_trace_dir = self.artifact_path / f"{prefix}_profile"
            output_trace_dir.mkdir(parents=True, exist_ok=True)

            def on_trace_ready(profile: torch.profiler.profile) -> None:
                logger.info(
                    profile.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=10
                    )
                )
                profile.export_chrome_trace(
                    str(output_trace_dir / f"trace-{profile.step_num}.json")
                )

            kwargs_handlers.append(
                ProfileKwargs(
                    activities=["cpu", "cuda"],
                    schedule_option={
                        "skip_first": 10,
                        "wait": 5,
                        "warmup": 1,
                        "active": 3,
                        "repeat": 2,
                    },
                    # with_stack=True,
                    on_trace_ready=on_trace_ready,
                )
            )

        logger.info(f"{torch.cuda.device_count()=}")
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            fullgraph = False
            torch.cuda.set_device(int(local_rank))
            kwargs_handlers.append(
                DistributedDataParallelKwargs(
                    comm_hook=DDPCommunicationHookType[mixed_precision.upper()]
                )
            )

        tensorboard_path = self.artifact_path / f"{prefix}_tensorboard"
        tensorboard_path.mkdir(parents=True, exist_ok=True)

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dynamo_plugin=TorchDynamoPlugin(
                backend="inductor",
                fullgraph=fullgraph,  # mode="reduce-overhead"
            ),
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=kwargs_handlers,
            log_with=TensorBoardTracker(
                run_name=str(time_ns()), logging_dir=tensorboard_path
            ),
        )

        if not self.is_profile:
            self.accelerator.profile = nullcontext

        logger.info(f"{self.accelerator.is_main_process=}")
        logger.info(f"{torch.cuda.get_device_properties(self.accelerator.device)=}")

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.datastore.name)

    def checkpoint(self, state: TrainingState) -> None:
        state_path = self.artifact_path / f"{self.checkpoint_prefix}{state.step_index}"
        self.accelerator.save_state(state_path)

    def train(
        self,
        # trial: optuna.trial.Trial,
        autoencoder: Model,
        discriminator: Model,
        train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        state = TrainingState()
        self.accelerator.register_for_checkpointing(state)

        checkpoints = list(self.artifact_path.glob(f"{self.checkpoint_prefix}*"))
        checkpoints.sort(
            key=lambda path: int(path.stem.removeprefix(self.checkpoint_prefix)),
        )
        latest_checkpoint = checkpoints[-1] if checkpoints else None

        if latest_checkpoint is not None:
            self.accelerator.load_state(latest_checkpoint)
            logger.info(f"Resuming from {state.epoch_index=} {state.step_index=}")

        def val_iterator() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            while True:
                yield from val_dataloader

        lpips: Any = LPIPS().to(self.accelerator.device)
        lpips = torch.compile(lpips, **self.accelerator.state.dynamo_plugin.to_kwargs())
        autoencoder_loss: Any = AutoencoderLoss(
            state=state,
            autoencoder=autoencoder,
            discriminator=discriminator,
            discriminator_warmup_steps=torch.tensor(self.discriminator_warmup_steps),
            lpips=lpips,
        ).to(self.accelerator.device)
        # autoencoder_loss = torch.compile(
        #     autoencoder_loss, **self.accelerator.state.dynamo_plugin.to_kwargs()
        # )

        for _ in range(self.epoch_count):
            epoch(
                state=state,
                accelerator=self.accelerator,
                autoencoder=autoencoder,
                autoencoder_loss=autoencoder_loss,
                discriminator=discriminator,
                train_dataloader=train_dataloader,
                checkpoint=partial(self.checkpoint, state=state),
                checkpoint_steps=self.checkpoint_steps,
                validate=partial(
                    validate,
                    state=state,
                    accelerator=self.accelerator,
                    model=autoencoder.model,
                    val_iterator=val_iterator(),
                    val_count=self.val_count,
                ),
                val_steps=self.val_steps,
                max_step_count=self.max_step_count,
            )
            state.epoch_index += 1

        self.accelerator.wait_for_everyone()
        self.checkpoint(state)

        self.accelerator.end_training()

    def run(self) -> None:
        channel_count = self.data_module_class.channel_count
        autoencoder_model = get_model(channel_count)
        logger.info(f"{autoencoder_model.num_parameters()=}")
        autoencoder_optimizer = bitsandbytes.optim.Lion(
            autoencoder_model.parameters(), optim_bits=32
        )
        discriminator_model = get_discriminator_model(channel_count)
        discriminator_optimizer = bitsandbytes.optim.AdamW(
            discriminator_model.parameters(), optim_bits=32
        )

        def save_model_hook(
            models: list[torch.nn.Module], weights: list[Any], output_dir: Path | str
        ) -> None:
            output_path = Path(output_dir)
            if self.accelerator.is_main_process:
                autoencoder_model, discriminator_model = models
                autoencoder_model.save_pretrained(output_path / "autoencoder")
                discriminator_model.save_pretrained(output_path / "discriminator")
                weights.pop()
                weights.pop()

        def load_model_hook(
            models: list[torch.nn.Module], input_dir: Path | str
        ) -> None:
            input_path = Path(input_dir)
            discriminator_model = models.pop()
            pretrained_discriminator_model = Discriminator.from_pretrained(
                input_path, subfolder="discriminator"
            )
            discriminator_model.register_to_config(
                **pretrained_discriminator_model.config
            )
            discriminator_model.load_state_dict(
                pretrained_discriminator_model.state_dict()
            )
            del pretrained_discriminator_model

            autoencoder_model = models.pop()
            pretrained_autoencoder_model = model_class.from_pretrained(
                input_path, subfolder="autoencoder"
            )
            autoencoder_model.register_to_config(**pretrained_autoencoder_model.config)
            autoencoder_model.load_state_dict(pretrained_autoencoder_model.state_dict())
            del pretrained_autoencoder_model

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        data_module = self.data_module_class(
            self.datastore,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            lengths=(0.8, 0.15, 0.05),
            image_size=image_size,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
        )
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()

        autoencoder = Model(
            autoencoder_model,
            autoencoder_optimizer,
            get_learning_rate_scheduler(
                autoencoder_optimizer, train_dataloader, self.epoch_count
            ),
        )
        discriminator = Model(
            discriminator_model,
            discriminator_optimizer,
            get_learning_rate_scheduler(
                discriminator_optimizer, train_dataloader, self.epoch_count
            ),
        )

        (
            autoencoder.model,
            autoencoder.optimizer,
            autoencoder.learning_rate_scheduler,
            discriminator.model,
            discriminator.optimizer,
            discriminator.learning_rate_scheduler,
            train_dataloader,
            val_dataloader,
        ) = self.accelerator.prepare(
            autoencoder.model,
            autoencoder.optimizer,
            autoencoder.learning_rate_scheduler,
            discriminator.model,
            discriminator.optimizer,
            discriminator.learning_rate_scheduler,
            train_dataloader,
            val_dataloader,
        )

        self.train(
            autoencoder,
            discriminator,
            train_dataloader,
            val_dataloader,
        )

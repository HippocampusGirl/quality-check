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
    # FP8RecipeKwargs,
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
from .loss import LPIPS, GradNorm, GradNormOutput
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
    autoencoder: Model[torch.nn.Module],
    discriminator: Model[torch.nn.Module],
    grad_norm: Model[GradNorm],
    lpips_loss: LPIPS,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    checkpoint: Callable[[], None],
    checkpoint_steps: int,
    validate: Callable[[], None],
    val_steps: int,
    log_steps: int = 50,
    max_step_count: int | None = None,
) -> None:
    writer = accelerator.get_tracker("tensorboard", unwrap=True)

    autoencoder.model = autoencoder.model.train()
    discriminator.model = discriminator.model.train()
    get_grad_norm_output = partial(
        grad_norm.model,
        state,
        autoencoder.model,
        discriminator.model,
        lpips_loss,
    )

    data_timer = Timer()
    model_timer = Timer()
    log_timer = Timer()

    data_timer.start()

    with (
        tqdm(
            total=len(train_dataloader),
            leave=False,
            mininterval=1.0,
            maxinterval=30.0,
            disable=not accelerator.is_main_process,
            unit="steps" + " ",
            position=0,
        ) as progress_bar,
        accelerator.profile() as profile,
    ):
        progress_bar.set_description(f"Epoch {state.epoch_index.item() + 1}")

        reconstructed_images: torch.Tensor | None = None
        for images, _ in islice(train_dataloader, max_step_count):
            images = VaeImageProcessor.normalize(images)
            data_timer.stop()

            model_timer.start()
            terms: dict[str, torch.Tensor] = dict()
            step = state.step_index.item() // accelerator.gradient_accumulation_steps
            is_autoencoder_step = (step % 2) == 0
            if is_autoencoder_step or reconstructed_images is None:
                with accelerator.accumulate(autoencoder.model):
                    o: GradNormOutput = get_grad_norm_output(images)
                    accelerator.backward(o.autoencoder_loss)
                    accelerator.backward(
                        o.grad_norm_loss, inputs=grad_norm.model.weights
                    )
                    reconstructed_images = o.reconstructed_images
                    keys = ["l1_loss", "lpips_loss", "jukebox_loss", "adversarial_loss"]
                    terms = dict(
                        autoencoder_loss=o.autoencoder_loss,
                        commit_loss=o.commit_loss,
                        **{
                            key: value
                            for key, value in zip(keys, o.values, strict=False)
                        },
                        **{
                            key.replace("loss", "weight"): weight
                            for key, weight in zip(keys, o.weights, strict=False)
                        },
                        grad_norm_loss=o.grad_norm_loss,
                    )

                    accelerator.clip_grad_norm_(autoencoder.model.parameters(), 1.0)
                    autoencoder.optimizer.step()
                    grad_norm.optimizer.step()
                    autoencoder.learning_rate_scheduler.step()
                    grad_norm.learning_rate_scheduler.step()
                autoencoder.optimizer.zero_grad(set_to_none=True)
                grad_norm.optimizer.zero_grad(set_to_none=True)
            else:
                with accelerator.accumulate(discriminator.model):
                    reconstructed_images = reconstructed_images.detach_()
                    reconstructed_score = discriminator.model(reconstructed_images)

                    images = images.requires_grad_()
                    score = discriminator.model(images)

                    hl = (
                        torch.nn.functional.relu(1.0 + reconstructed_score)
                        + torch.nn.functional.relu(1.0 - score)
                    ).mean()
                    gp = gradient_penalty(images, score)

                    loss = hl + gp
                    terms = dict(
                        discriminator_hinge_loss=hl,
                        discriminator_gradient_penalty=gp,
                        discriminator_loss=loss,
                    )

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(discriminator.model.parameters(), 1.0)
                    discriminator.optimizer.step()
                    discriminator.learning_rate_scheduler.step()
                discriminator.optimizer.zero_grad(set_to_none=True)

            model_timer.stop()

            if progress_bar.n < 10:
                model_timer.reset()
                data_timer.reset()
                log_timer.reset()

            progress_bar.update(1)
            if step != 0:
                if (step % log_steps) in {0, 1}:
                    log_timer.start()
                    terms = accelerator.reduce(terms, reduction="mean")
                    if accelerator.sync_gradients and accelerator.is_main_process:
                        learning_rate = (
                            autoencoder.learning_rate_scheduler.get_last_lr()[0]
                        )
                        logs = dict(
                            learning_rate=learning_rate,
                            epoch=state.epoch_index.item(),
                            data_time=data_timer.value,
                            model_time=model_timer.value,
                            log_time=log_timer.value,
                        )
                        progress_bar.set_postfix(**logs, refresh=False)
                        logs.update(
                            {
                                f"train_{key}": value.item()
                                for key, value in terms.items()
                            }
                        )
                        for key, value in logs.items():
                            writer.add_scalar(key, value, global_step=step)
                    log_timer.stop()

                if (
                    (step % checkpoint_steps) == 0
                    and accelerator.sync_gradients
                    and accelerator.is_main_process
                ):
                    checkpoint()
                if (step % val_steps) == 0:
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

    val_steps: int = 200
    val_count: int = 50
    checkpoint_steps: int = 1000
    # https://github.com/CompVis/taming-transformers/issues/93
    discriminator_warmup_steps: int = 30000

    artifact_path: Path = field(init=False)
    accelerator: Accelerator = field(init=False)
    checkpoint_prefix: str = field(init=False)

    def __post_init__(self) -> None:
        if self.datastore.cache_path is None:
            raise ValueError("Datastore cache path is None")
        prefix = f"data-{self.data_module_class.name}_model-{model_name}"
        self.checkpoint_prefix = f"{prefix}_step-"
        self.artifact_path = self.datastore.cache_path

        kwargs_handlers: list[Any] = list()
        fsdp_plugin: FullyShardedDataParallelPlugin | None = None

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

        fullgraph: bool = True
        torch._dynamo.config.compiled_autograd = True

        logger.info(f"{torch.cuda.device_count()=}")
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            fullgraph = False
            torch.cuda.set_device(int(local_rank))
            kwargs_handlers.append(
                DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
            )

        mixed_precision = "bf16"
        # mixed_precision = "fp8"
        # if mixed_precision == "fp8":
        #     fullgraph = False
        #     fp8_recipe_kwargs = FP8RecipeKwargs(backend="te", fp8_format="HYBRID")
        #     kwargs_handlers.append(fp8_recipe_kwargs)

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
                run_name=str(time_ns()), logging_dir=tensorboard_path, max_queue=0
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
        state: TrainingState,
        autoencoder: Model[torch.nn.Module],
        discriminator: Model[torch.nn.Module],
        grad_norm: Model[GradNorm],
        train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        checkpoints = list(self.artifact_path.glob(f"{self.checkpoint_prefix}*"))
        checkpoints.sort(
            key=lambda path: int(path.stem.removeprefix(self.checkpoint_prefix)),
        )
        latest_checkpoint = checkpoints[-1] if checkpoints else None

        if latest_checkpoint is not None:
            self.accelerator.load_state(latest_checkpoint)
            logger.info(f"Resuming from {state.epoch_index=} {state.step_index=}")

        lpips_loss: Any = LPIPS().to(self.accelerator.device)
        lpips_loss = torch.compile(
            lpips_loss, **self.accelerator.state.dynamo_plugin.to_kwargs()
        )

        def val_iterator() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            while True:
                yield from val_dataloader

        for _ in range(self.epoch_count):
            epoch(
                state=state,
                accelerator=self.accelerator,
                autoencoder=autoencoder,
                discriminator=discriminator,
                lpips_loss=lpips_loss,
                grad_norm=grad_norm,
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
        state = TrainingState()
        self.accelerator.register_for_checkpointing(state)

        channel_count = self.data_module_class.channel_count

        autoencoder_model = get_model(channel_count)
        autoencoder_optimizer = bitsandbytes.optim.Lion(
            autoencoder_model.parameters(), optim_bits=32
        )

        discriminator_model = get_discriminator_model(channel_count)
        discriminator_optimizer = bitsandbytes.optim.AdamW(
            discriminator_model.parameters(), optim_bits=32
        )

        grad_norm_model = GradNorm(
            discriminator_warmup_steps=self.discriminator_warmup_steps
        )
        self.accelerator.register_for_checkpointing(grad_norm_model)
        grad_norm_optimizer = bitsandbytes.optim.AdamW(
            grad_norm_model.parameters(), optim_bits=32
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

        logger.info(f"{autoencoder_model.num_parameters()=}")

        learning_rate_scheduler_factory = partial(
            get_learning_rate_scheduler,
            train_dataloader=train_dataloader,
            epoch_count=self.epoch_count,
        )

        autoencoder = Model(
            autoencoder_model,
            autoencoder_optimizer,
            learning_rate_scheduler_factory(autoencoder_optimizer),
        )
        discriminator = Model(
            discriminator_model,
            discriminator_optimizer,
            learning_rate_scheduler_factory(discriminator_optimizer),
        )
        grad_norm = Model(
            grad_norm_model.to(self.accelerator.device),
            grad_norm_optimizer,
            learning_rate_scheduler_factory(grad_norm_optimizer),
        )

        (
            autoencoder.model,
            autoencoder.optimizer,
            autoencoder.learning_rate_scheduler,
            discriminator.model,
            discriminator.optimizer,
            discriminator.learning_rate_scheduler,
            # grad_norm.model,
            grad_norm.optimizer,
            grad_norm.learning_rate_scheduler,
            train_dataloader,
            val_dataloader,
        ) = self.accelerator.prepare(
            autoencoder.model,
            autoencoder.optimizer,
            autoencoder.learning_rate_scheduler,
            discriminator.model,
            discriminator.optimizer,
            discriminator.learning_rate_scheduler,
            # grad_norm.model,
            grad_norm.optimizer,
            grad_norm.learning_rate_scheduler,
            train_dataloader,
            val_dataloader,
        )

        self.train(
            state,
            autoencoder,
            discriminator,
            grad_norm,
            train_dataloader,
            val_dataloader,
        )

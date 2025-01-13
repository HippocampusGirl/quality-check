import os
from dataclasses import dataclass, field
from typing import Any, Type

import bitsandbytes
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DDPCommunicationHookType,
    DistributedDataParallelKwargs,
    FullyShardedDataParallelPlugin,
    TorchDynamoPlugin,
)
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.datamodule import DataModule
from ..data.schema import Datastore
from ..logging import logger
from ..utils import Timer
from .evaluate import evaluate
from .model import (
    get_discriminator_model,
    get_learning_rate_scheduler,
    get_model,
)


@dataclass
class TrainingState:
    epoch_index: int = 0
    step_index: int = 0


@dataclass
class Model:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rate_scheduler": self.learning_rate_scheduler.state_dict(),  # type: ignore
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.learning_rate_scheduler.load_state_dict(
            state_dict["learning_rate_scheduler"]
        )  # type: ignore


def epoch(
    state: TrainingState,
    accelerator: Accelerator,
    autoencoder: Model,
    discriminator: Model,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    autoencoder.model = autoencoder.model.train()
    discriminator.model = discriminator.model.train()

    data_timer = Timer()
    model_timer = Timer()

    data_timer.start()

    reconstructed_images: torch.Tensor | None = None
    with tqdm(total=len(train_dataloader), leave=False) as progress_bar:
        progress_bar.set_description(f"Epoch {state.epoch_index + 1}")
        for images, _ in train_dataloader:
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
                    loss = mse_loss(images, reconstructed_images)
                    loss += commit_loss

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(autoencoder.model.parameters(), 1.0)
                    autoencoder.optimizer.step()
                    autoencoder.learning_rate_scheduler.step()
                autoencoder.optimizer.zero_grad(set_to_none=True)
            else:
                with accelerator.accumulate(discriminator.model):
                    reconstructed_images = reconstructed_images.detach_()
                    reconstructed = discriminator.model(reconstructed_images)

                    images = images.requires_grad_()
                    real = discriminator.model(images)

                    loss = (
                        torch.nn.functional.relu(1 + reconstructed)
                        + torch.nn.functional.relu(1 - real)
                    ).mean()
                    # loss += gradient_penalty(images, real)

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(discriminator.model.parameters(), 1.0)
                    discriminator.optimizer.step()
                    discriminator.learning_rate_scheduler.step()
                discriminator.optimizer.zero_grad(set_to_none=True)

            model_timer.stop()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "learning_rate": autoencoder.learning_rate_scheduler.get_last_lr()[0],
                "step": state.step_index,
                "data_time": data_timer.value,
                "model_time": model_timer.value,
            }
            progress_bar.set_postfix(**logs)
            state.step_index += 1

            data_timer.start()


@dataclass
class Trainer:
    datastore: Datastore
    data_module_class: Type[DataModule]

    train_batch_size: int
    gradient_accumulation_steps: int
    eval_batch_size: int

    seed: int

    epoch_count: int

    accelerator: Accelerator = field(init=False)

    def __post_init__(self) -> None:
        fullgraph = True
        mixed_precision = "bf16"
        kwargs_handlers: list[Any] = list()
        # mixed_precision = "fp8"
        # if mixed_precision == "fp8":
        #     fullgraph = False
        #     fp8_recipe_kwargs = FP8RecipeKwargs(backend="te", fp8_format="HYBRID")
        #     kwargs_handlers.append(fp8_recipe_kwargs)
        fsdp_plugin: FullyShardedDataParallelPlugin | None = None
        # deepspeed_plugin: DeepSpeedPlugin

        cuda_device_count = torch.cuda.device_count()
        logger.info(f"{torch.cuda.device_count()=}")
        if cuda_device_count > 1:
            fullgraph = False
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            kwargs_handlers.append(
                DistributedDataParallelKwargs(
                    comm_hook=DDPCommunicationHookType[mixed_precision.upper()]
                )
            )
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dynamo_plugin=TorchDynamoPlugin(
                backend="inductor",
                fullgraph=fullgraph,  # , mode="reduce-overhead"
            ),
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=kwargs_handlers,
        )

        torch._dynamo.config.cache_size_limit = 64
        torch.set_float32_matmul_precision("high")

        logger.info(f"{self.accelerator.is_main_process=}")
        logger.info(f"{torch.cuda.get_device_properties(self.accelerator.device)=}")

    def train_autoencoder(
        self,
        # trial: optuna.trial.Trial,
        autoencoder: Model,
        discriminator: Model,
        train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        loss = float("inf")

        # study = trial.study
        # artifact_id = None
        # retry_history = RetryFailedTrialCallback.retry_history(trial)  # type: ignore
        # for trial_number in reversed(retry_history):
        #     artifact_id = study.trials[trial_number].user_attrs.get("artifact_id")
        #     if artifact_id is not None:
        #         retry_trial_number = trial_number
        #         break

        # if artifact_id is not None:
        #     with TemporaryDirectory() as temporary_directory_str:
        #         path = Path(temporary_directory_str) / "model.pt"
        #         download_artifact(
        #             artifact_store=self.artifact_store,
        #             file_path=str(path),
        #             artifact_id=artifact_id,
        #         )
        #         checkpoint = torch.load(path, weights_only=True)
        #     state = TrainingState(**checkpoint["training_state_dict"])
        #     state.epoch_index += 1

        #     logger.info(
        #         f"Resuming from trial {retry_trial_number} in epoch
        # {state.epoch_index}"
        #     )

        #     autoencoder.load_state_dict(checkpoint["autoencoder"])
        #     discriminator.load_state_dict(checkpoint["discriminator"])
        #     loss = checkpoint["loss"]
        # else:
        state = TrainingState()

        for epoch_index in range(state.epoch_index, self.epoch_count):
            state.epoch_index = epoch_index

            epoch(
                state,
                self.accelerator,
                autoencoder,
                discriminator,
                train_dataloader,
            )

            if (
                self.accelerator.is_main_process
                and self.datastore.cache_path is not None
            ):
                self.accelerator.save_state(
                    self.datastore.cache_path
                    / f"model-autoencoder_epoch-{state.epoch_index}.pt"
                )

            loss = evaluate(
                autoencoder.model,
                val_dataloader,
                step_count=50,
                step_pruning=5,
            )
            # trial.report(loss, state.step_index)
            logger.info(f"Reported {loss=} at {state.step_index=}")

        return loss

    def objective(self) -> float:
        channel_count = self.data_module_class.channel_count
        autoencoder_model = get_model(channel_count)
        autoencoder_optimizer = bitsandbytes.optim.AdamW8bit(
            autoencoder_model.parameters(), lr=1e-4
        )
        discriminator_model = get_discriminator_model(channel_count)
        discriminator_optimizer = bitsandbytes.optim.AdamW8bit(
            discriminator_model.parameters(), lr=1e-4
        )

        data_module = self.data_module_class(
            self.datastore,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            lengths=(0.8, 0.15, 0.05),
            image_size=512,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
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

        return self.train_autoencoder(
            autoencoder,
            discriminator,
            train_dataloader,
            val_dataloader,
        )

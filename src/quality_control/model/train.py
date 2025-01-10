import os
import signal
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Type

import optuna
import torch
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
from diffusers import DDPMScheduler
from optuna.artifacts import FileSystemArtifactStore, download_artifact, upload_artifact
from optuna.storages import RetryFailedTrialCallback
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.datamodule import DataModule
from ..data.schema import Datastore
from ..logging import logger
from .evaluate import evaluate
from .unet import get_learning_rate_scheduler, get_model, get_optimizer


@dataclass
class TrainingState:
    epoch_index: int = 0
    step_index: int = 0


def epoch(
    state: TrainingState,
    accelerator: Accelerator,
    generator: torch.Generator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    timestep_count: int,
) -> None:
    device = model.device
    noise_scheduler = DDPMScheduler(num_train_timesteps=timestep_count)
    with tqdm(total=len(train_dataloader), leave=False) as progress_bar:
        progress_bar.set_description(f"Epoch {state.epoch_index + 1}")
        for clean_images, class_labels in train_dataloader:
            # Sample noise to add to the images
            noise = torch.empty_like(clean_images).normal_(generator=generator)
            batch_size = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                timestep_count,
                (batch_size,),
                dtype=torch.int64,
                device=device,
                generator=generator,
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)  # type: ignore

            with accelerator.accumulate(model):
                # Predict the noise residual
                (predicted_noise,) = model(
                    noisy_images,
                    timesteps,
                    class_labels=class_labels,
                    return_dict=False,
                )
                loss = mse_loss(predicted_noise, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                learning_rate_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # if step_index == 0:  # Reset timers
            #     progress_bar.start_t = progress_bar._time()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "learning_rate": learning_rate_scheduler.get_last_lr()[0],
                "step": state.step_index,
            }
            progress_bar.set_postfix(**logs)
            state.step_index += 1

            # rate = progress_bar.format_dict["rate"]
            # remaining = (progress_bar.total - progress_bar.n) / rate
            # if remaining > 3600 and step_index > 2:
            #     raise torch.OutOfMemoryError  # Too slow


@dataclass
class Trainer:
    datastore: Datastore
    artifact_store: FileSystemArtifactStore
    data_module_class: Type[DataModule]

    train_batch_size: int
    gradient_accumulation_steps: int
    eval_batch_size: int

    seed: int

    timestep_count: int
    epoch_count: int

    accelerator: Accelerator = field(init=False)

    def __post_init__(self) -> None:
        if torch.cuda.get_device_capability() >= (8, 0):
            mixed_precision = "bf16"
        else:
            mixed_precision = "fp16"
        # mixed_precision = "fp8"
        # fp8_recipe_kwargs = FP8RecipeKwargs(backend="msamp", opt_level="02")
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dynamo_plugin=TorchDynamoPlugin(
                backend="inductor", fullgraph=True, mode="reduce-overhead"
            ),
            # kwargs_handlers=[fp8_recipe_kwargs],
        )

        torch._dynamo.config.cache_size_limit = 64

        device = self.accelerator.device
        logger.info(f"{device=}")
        logger.info(f"{torch.cuda.get_device_capability(device)=}")

    @property
    def device(self) -> Any:
        return self.accelerator.device

    def train(
        self,
        trial: optuna.trial.Trial,
        generator: torch.Generator,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR,
        train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        r2 = 0.0

        study = trial.study
        artifact_id = None
        retry_history = RetryFailedTrialCallback.retry_history(trial)  # type: ignore
        for trial_number in reversed(retry_history):
            artifact_id = study.trials[trial_number].user_attrs.get("artifact_id")
            if artifact_id is not None:
                retry_trial_number = trial_number
                break

        if artifact_id is not None:
            with TemporaryDirectory() as temporary_directory_str:
                path = Path(temporary_directory_str) / "model.pt"
                download_artifact(
                    artifact_store=self.artifact_store,
                    file_path=str(path),
                    artifact_id=artifact_id,
                )
                checkpoint = torch.load(path, weights_only=True)
            state = TrainingState(**checkpoint["training_state_dict"])
            state.epoch_index += 1

            logger.info(
                f"Resuming model {model} from trial {retry_trial_number} "
                f"in epoch {state.epoch_index}"
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            r2 = checkpoint["r2"]
        else:
            logger.info(f"Training model {model}")
            state = TrainingState()

        for epoch_index in range(state.epoch_index, self.epoch_count):
            state.epoch_index = epoch_index

            epoch(
                state,
                self.accelerator,
                generator,
                model,
                optimizer,
                learning_rate_scheduler,
                train_dataloader,
                self.timestep_count,
            )
            optimizer.zero_grad(set_to_none=True)

            r2 = evaluate(
                model,
                val_dataloader,
                step_count=50,
                step_pruning=5,
            )
            trial.report(r2, state.step_index)
            logger.info(f"Reported {r2=} at {state.step_index=}")

            with TemporaryDirectory() as temporary_directory_str:
                path = Path(temporary_directory_str) / "model.pt"
                torch.save(
                    {
                        "training_state_dict": asdict(state),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "r2": r2,
                    },
                    path,
                )
                artifact_id = upload_artifact(
                    artifact_store=self.artifact_store,
                    file_path=str(path),
                    study_or_trial=trial,
                )
            trial.set_user_attr("artifact_id", artifact_id)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return r2

    def _objective(self, trial: optuna.trial.Trial) -> float:
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        channel_count = self.data_module_class.channel_count
        class_count = self.data_module_class.class_count
        model = get_model(trial, channel_count, class_count)
        parameter_count = model.num_parameters()
        memory_footprint: float = model.get_memory_footprint(return_buffers=True) / 1e9
        logger.info(f"{memory_footprint=} {parameter_count=}")
        # with torch.device(self.device):
        optimizer = get_optimizer(trial, model)

        data_module = self.data_module_class(
            self.datastore,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            lengths=(0.8, 0.15, 0.05),
            image_size=model.sample_size,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
        )
        train_dataloader = data_module.train_dataloader()
        learning_rate_scheduler = get_learning_rate_scheduler(
            optimizer, train_dataloader, self.epoch_count
        )
        val_dataloader = data_module.val_dataloader()

        (
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            learning_rate_scheduler,
        ) = self.accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            learning_rate_scheduler,
        )

        return self.train(
            trial,
            generator,
            model,
            optimizer,
            learning_rate_scheduler,
            train_dataloader,
            val_dataloader,
        )

    def objective(self, trial: optuna.trial.Trial) -> float:
        try:
            r2 = self._objective(trial)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            logger.info("Out of memory", exc_info=e, stack_info=False)
            trial.set_user_attr("constraint", (1,))
            return -1.0
        except KeyboardInterrupt:
            # Exit immediately without letting optuna mark the trial as failed
            os.kill(os.getpid(), signal.SIGKILL)
        self.accelerator.free_memory()
        return r2


def constraints(trial: optuna.trial.Trial) -> Any:
    return trial.user_attrs["constraint"]

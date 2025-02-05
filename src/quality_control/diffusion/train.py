import os
import signal
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path
from shutil import make_archive, unpack_archive
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator, Type

import bitsandbytes
import optuna
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DDPCommunicationHookType,
    DistributedDataParallelKwargs,
    FP8RecipeKwargs,
    FullyShardedDataParallelPlugin,
    ProfileKwargs,
    TorchDynamoPlugin,
)
from diffusers import DDPMScheduler
from optuna.artifacts import FileSystemArtifactStore, download_artifact, upload_artifact
from optuna.storages import RetryFailedTrialCallback
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..autoencoder.loss import LPIPS
from ..autoencoder.model import image_size
from ..data.datamodule import DataModule
from ..data.schema import Datastore
from ..logging import logger
from ..utils import Timer, TrainingState
from .model import get_learning_rate_scheduler, get_model
from .validate import validate

torch._dynamo.config.cache_size_limit = 1 << 10
torch.set_float32_matmul_precision("high")
# torch.multiprocessing.set_sharing_strategy("file_system")  # type: ignore


def epoch(
    state: TrainingState,
    accelerator: Accelerator,
    generator: torch.Generator,
    diffusion_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    timestep_count: int,
    log_steps: int,
    checkpoint: Callable[[], None],
    checkpoint_steps: int,
    validate: Callable[[], float],
    val_steps: int,
    # timeout: Timeout,
    max_step_count: int | None = None,
) -> float:
    r2: float = 0.0

    diffusion_model = diffusion_model.train()

    data_timer = Timer()
    model_timer = Timer()

    data_timer.start()

    noise_scheduler = DDPMScheduler(num_train_timesteps=timestep_count)
    with (
        tqdm(
            total=len(train_dataloader),
            leave=False,
            disable=not accelerator.is_main_process,
            unit="steps" + " ",
            position=1,
        ) as progress_bar,
        accelerator.profile() as profile,
    ):
        for clean_latents, class_labels in islice(train_dataloader, max_step_count):
            with torch.no_grad():
                batch_size = clean_latents.size(0)
                # Sample noise to add to the images
                noise = torch.empty_like(clean_latents).normal_(generator=generator)
                timesteps = torch.randint(
                    0,
                    timestep_count,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                    generator=generator,
                )
                noisy_latents = noise_scheduler.add_noise(  # type: ignore
                    clean_latents, noise, timesteps
                )

            data_timer.stop()
            model_timer.start()

            with accelerator.accumulate(diffusion_model):
                # Predict the noise residual
                (predicted_noise,) = diffusion_model(
                    noisy_latents,
                    timesteps,
                    class_labels=class_labels,
                    return_dict=False,
                )
                loss = mse_loss(predicted_noise, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(diffusion_model.parameters(), 1.0)
                optimizer.step()
                learning_rate_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            model_timer.stop()
            # timeout.reset(600)

            if progress_bar.n < 10:  # Reset timers after warmup
                progress_bar.start_t = progress_bar._time()
                model_timer.reset()
                data_timer.reset()

            progress_bar.update(1)
            if state.step_index != 0:
                if state.step_index % log_steps == 0:
                    learning_rate = learning_rate_scheduler.get_last_lr()[0]
                    logs = dict(
                        learning_rate=learning_rate,
                        epoch=state.epoch_index.item(),
                        data_time=data_timer.value,
                        model_time=model_timer.value,
                        train_loss=loss.item(),
                    )
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=state.step_index.item())

                    if progress_bar.n > val_steps // 2:
                        time = progress_bar.total * model_timer.value
                        if time > 3600:
                            raise TimeoutError
                if state.step_index % checkpoint_steps == 0:
                    checkpoint()
                if state.step_index % val_steps == 0:
                    r2 = validate()
                # else:
                #     timeout.reset(10)

            state.step_index += 1

            if profile is not None:
                profile.step()

            data_timer.start()

    return r2


@dataclass
class Trainer:
    datastore: Datastore
    data_module_class: Type[DataModule]
    seed: int

    autoencoder_model: Any

    artifact_store: FileSystemArtifactStore

    batch_size: int
    gradient_accumulation_steps: int

    epoch_count: int
    max_step_count: int | None = None

    timestep_count: int = 1000

    is_profile: bool = False

    log_steps: int = 10
    val_steps: int = 500
    val_count: int = 20
    val_timestep_count: int = 50
    val_timestep_pruning: int = 5
    checkpoint_steps: int = 1000

    accelerator: Accelerator = field(init=False)
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = field(init=False)
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        init=False
    )

    def __post_init__(self) -> None:
        data_module = self.data_module_class(
            self.datastore,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            lengths=(0.8, 0.15, 0.05),
            image_size=image_size,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            autoencoder_model=self.autoencoder_model,
        )
        self.train_dataloader = data_module.train_dataloader()
        self.val_dataloader = data_module.val_dataloader()

        if self.datastore.cache_path is None:
            raise ValueError("Datastore cache path is None")
        self.artifact_path = self.datastore.cache_path

        kwargs_handlers: list[Any] = list()

        if self.is_profile:
            prefix = f"dataset-{self.datastore.name}_model-diffusion"
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
                    with_stack=True,
                    on_trace_ready=on_trace_ready,
                )
            )

        fullgraph: bool = True
        mixed_precision: str = "bf16"
        if mixed_precision == "fp8":
            fullgraph = False
            fp8_recipe_kwargs = FP8RecipeKwargs(backend="te", fp8_format="HYBRID")
            kwargs_handlers.append(fp8_recipe_kwargs)
        fsdp_plugin: FullyShardedDataParallelPlugin | None = None

        cuda_device_count = torch.cuda.device_count()
        logger.info(f"{torch.cuda.device_count()=}")
        if cuda_device_count > 1:
            fullgraph = False
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            kwargs_handlers.append(
                DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
            )

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dynamo_plugin=TorchDynamoPlugin(
                backend="inductor", fullgraph=fullgraph, mode="reduce-overhead"
            ),
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=kwargs_handlers,
        )

        if not self.is_profile:
            self.accelerator.profile = nullcontext

    def checkpoint(self, trial: optuna.trial.Trial) -> None:
        with TemporaryDirectory() as temporary_directory_str:
            temporary_directory = Path(temporary_directory_str)
            model_path = temporary_directory / "model"
            self.accelerator.save_state(str(model_path))
            file_path = make_archive(str(model_path), "zip", model_path)
            artifact_id = upload_artifact(
                artifact_store=self.artifact_store,
                file_path=file_path,
                study_or_trial=trial,
            )
        trial.set_user_attr("artifact_id", artifact_id)

    def train(
        self,
        trial: optuna.trial.Trial,
        generator: torch.Generator,
        diffusion_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR,
        train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        prefix = (
            f"dataset-{self.datastore.name}_model-diffusion_"
            f"study-{trial.study.study_name}"
        )
        tensorboard_path = self.artifact_path / f"{prefix}_tensorboard"
        tensorboard_path.mkdir(parents=True, exist_ok=True)
        if len(self.accelerator.log_with) == 0:
            self.accelerator.log_with.append("tensorboard")
        self.accelerator.project_configuration.set_directories(str(tensorboard_path))
        self.accelerator.init_trackers(f"trial-{trial.number}")

        state = TrainingState()
        self.accelerator.register_for_checkpointing(state)

        study = trial.study
        artifact_id = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retry_history = RetryFailedTrialCallback.retry_history(trial)  # type: ignore
        for trial_number in reversed(retry_history):
            artifact_id = study.trials[trial_number].user_attrs.get("artifact_id")
            if artifact_id is not None:
                retry_trial_number = trial_number
                break

        if artifact_id is not None:
            with TemporaryDirectory() as temporary_directory_str:
                temporary_directory = Path(temporary_directory_str)
                model_path = temporary_directory / "model.zip"
                download_artifact(
                    artifact_store=self.artifact_store,
                    file_path=str(model_path),
                    artifact_id=artifact_id,
                )
                model_directory = temporary_directory / "model"
                unpack_archive(model_path, extract_dir=model_directory)
                self.accelerator.load_state(str(model_directory))
                logger.info(
                    f"Resuming from trial {retry_trial_number} "
                    f"{state.epoch_index=} {state.step_index=}"
                )

        def val_iterator() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            while True:
                yield from val_dataloader

        lpips: Any = LPIPS().to(self.accelerator.device)
        lpips = torch.compile(lpips, **self.accelerator.state.dynamo_plugin.to_kwargs())

        epochs = list(range(int(state.epoch_index.item()), self.epoch_count))
        for _ in tqdm(
            epochs,
            leave=False,
            disable=not self.accelerator.is_main_process,
            unit="epochs" + " ",
            position=0,
        ):
            # with Timeout(seconds=300) as timeout:
            r2 = epoch(
                state=state,
                accelerator=self.accelerator,
                generator=generator,
                diffusion_model=diffusion_model,
                optimizer=optimizer,
                learning_rate_scheduler=learning_rate_scheduler,
                timestep_count=self.timestep_count,
                train_dataloader=train_dataloader,
                checkpoint=partial(self.checkpoint, trial=trial),
                checkpoint_steps=self.checkpoint_steps,
                validate=partial(
                    validate,
                    trial=trial,
                    state=state,
                    accelerator=self.accelerator,
                    autoencoder_model=self.autoencoder_model,
                    diffusion_model=diffusion_model,
                    lpips=lpips,
                    val_iterator=val_iterator(),
                    val_count=self.val_count,
                    val_timestep_count=self.val_timestep_count,
                    val_timestep_pruning=self.val_timestep_pruning,
                ),
                log_steps=self.log_steps,
                val_steps=self.val_steps,
                # timeout=timeout,
                max_step_count=self.max_step_count,
            )
            state.epoch_index += 1

        self.checkpoint(trial)
        self.accelerator.end_training()
        self.accelerator.trackers.clear()

        return r2

    def _objective(self, trial: optuna.trial.Trial) -> float:
        generator = torch.Generator(device=self.accelerator.device).manual_seed(
            self.seed
        )

        channel_count = self.autoencoder_model.config["latent_channels"]
        factor = 2 ** (len(self.autoencoder_model.config["down_block_types"]) - 1)
        class_count = self.data_module_class.class_count
        diffusion_model = get_model(
            trial, image_size // factor, channel_count, class_count
        )
        parameter_count = diffusion_model.num_parameters()  # type: ignore
        memory_footprint: float = (
            diffusion_model.get_memory_footprint(return_buffers=True) / 1e9  # type: ignore
        )
        logger.info(f"diffusion_model {memory_footprint=} {parameter_count=}")
        if memory_footprint > 10:
            trial.set_user_attr("constraint", (0.0, 1.0))
            return -1.0
        optimizer = bitsandbytes.optim.Lion(diffusion_model.parameters(), optim_bits=32)
        learning_rate_scheduler = get_learning_rate_scheduler(
            optimizer, self.train_dataloader, self.epoch_count
        )

        (
            diffusion_model,
            optimizer,
            train_dataloader,
            val_dataloader,
            learning_rate_scheduler,
        ) = self.accelerator.prepare(
            diffusion_model,
            optimizer,
            self.train_dataloader,
            self.val_dataloader,
            learning_rate_scheduler,
        )

        return self.train(
            trial,
            generator,
            diffusion_model,
            optimizer,
            learning_rate_scheduler,
            train_dataloader,
            val_dataloader,
        )

    def objective(self, trial: optuna.trial.Trial) -> float:
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)
            r2 = self._objective(trial)
        except (torch.OutOfMemoryError, RuntimeError, TimeoutError) as e:
            logger.info(
                "Marking trial as constrained after error",
                exc_info=e,
                stack_info=False,
            )
            trial.set_user_attr("constraint", (1.0, 0.0))
            return -1.0
        except KeyboardInterrupt as e:
            logger.error(
                "Received KeyboardInterrupt",
                exc_info=e,
                stack_info=False,
            )
            # Exit immediately without letting optuna mark the trial as failed
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0)
            self.accelerator.free_memory()

        return r2


def constraints(trial: optuna.trial.Trial) -> Any:
    return trial.user_attrs["constraint"]

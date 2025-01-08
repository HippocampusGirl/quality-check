from dataclasses import dataclass
from typing import Type

import numpy as np
import optuna
import scipy
import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler, PNDMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.datamodule import DataModule
from ..data.schema import Datastore

block_types = [
    "{direction}Block2D",
    "Attn{direction}Block2D",
    # "Skip{direction}Block2D",
    # "AttnSkip{direction}Block2D",
    # "K{direction}Block2D",
]

epoch_count = 300


def get_model(
    trial: optuna.trial.Trial,
    channel_count: int,
    class_count: int,
) -> torch.nn.Module:
    block_count = trial.suggest_int("block_count", 1, 6)
    down_block_types: list[str] = list()
    up_block_types: list[str] = list()
    for i in range(block_count):
        block_type = trial.suggest_categorical(f"block_type_{i}", block_types)
        down_block_types.append(block_type.format(direction="Down"))
        up_block_types.append(block_type.format(direction="Up"))

    norm_group_count = 32
    block_out_channels_base = 32 * trial.suggest_int(
        "block_out_channels_factor", 32 // norm_group_count, 512 // norm_group_count
    )
    block_out_channels: list[int] = [block_out_channels_base]
    block_out_channels_factor = 1
    for i in range(1, block_count):
        block_out_channels_factor += trial.suggest_int(
            f"block_out_channels_factor_increment_{i}", 0, 1
        )
        block_out_channels.append(block_out_channels[0] * block_out_channels_factor)

    downsample_type = trial.suggest_categorical("downsample_type", ["conv", "resnet"])
    upsample_type = trial.suggest_categorical("upsample_type", ["conv", "resnet"])

    image_size_base = 2 ** (block_count - 1)
    image_size = image_size_base * trial.suggest_int(
        "image_size_factor", 128 // image_size_base, 768 // image_size_base
    )

    model = UNet2DModel(
        sample_size=image_size,
        in_channels=channel_count,
        out_channels=channel_count,
        center_input_sample=True,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=trial.suggest_int("layers_per_block", 1, 3),
        downsample_type=downsample_type,
        upsample_type=upsample_type,
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        norm_num_groups=norm_group_count,
        num_class_embeds=class_count,
    )
    print(model)
    return torch.compile(model)  # type: ignore


def get_optimizer(
    trial: optuna.trial.Trial, model: torch.nn.Module
) -> torch.optim.Optimizer:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer


def get_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> torch.optim.lr_scheduler.LambdaLR:
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * epoch_count),
    )


def train(
    trial: optuna.trial.Trial,
    generator: torch.Generator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    epoch_count: int,
    timestep_count: int,
) -> float:
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        # log_with="tensorboard",
    )
    model, optimizer, train_dataloader, learning_rate_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, learning_rate_scheduler
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=timestep_count)

    r2 = 0.0
    step_index = 0

    for epoch_index in range(epoch_count):
        with tqdm(total=len(train_dataloader), leave=False) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch_index + 1}")
            for clean_images, class_labels in train_dataloader:
                # Sample noise to add to the images
                noise = torch.empty_like(clean_images).normal_(generator=generator)
                batch_size = clean_images.shape[0]

                timesteps = torch.randint(
                    0,
                    timestep_count,
                    (batch_size,),
                    dtype=torch.int64,
                    device=clean_images.device,
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
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": learning_rate_scheduler.get_last_lr()[0],
                    "step": step_index,
                }
                progress_bar.set_postfix(**logs)
                step_index += 1

        r2 = evaluate(model, val_dataloader, 50)
        trial.report(r2, step_index)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return r2


def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    step_count: int,
) -> float:
    device = model.device

    noise_scheduler = PNDMScheduler()
    noise_scheduler.set_timesteps(num_inference_steps=step_count)  # type: ignore
    timesteps = noise_scheduler.timesteps  # type: ignore

    good_bad_labels = torch.zeros(0, dtype=torch.int64, device=device)
    loss_values = torch.zeros(0, dtype=torch.float32, device=device)

    for clean_images, class_labels, _good_bad_labels in val_dataloader:
        _loss_values = torch.zeros_like(_good_bad_labels, dtype=torch.float32)
        for start in timesteps[1:]:
            # Sample noise to add to the images
            noise = torch.randn_like(clean_images)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, start)  # type: ignore

            for step in timesteps[timesteps <= start]:
                (predicted_noise,) = model(
                    noisy_images,
                    step,
                    class_labels=class_labels,
                    return_dict=False,
                )
                (noisy_images,) = noise_scheduler.step(  # type: ignore
                    predicted_noise, step, noisy_images, return_dict=False
                )

            denoised_images = (noisy_images / 2 + 0.5).clamp(0, 1)
            _loss_values += mse_loss(
                denoised_images, clean_images, reduction="none"
            ).mean(dim=(1, 2, 3))

        good_bad_labels = torch.cat((good_bad_labels, _good_bad_labels), dim=0)
        loss_values = torch.cat((loss_values, _loss_values), dim=0)

    # T-test for difference by label
    r = scipy.stats.pearsonr(good_bad_labels.cpu().numpy(), loss_values.cpu().numpy()).r
    return float(np.square(r))


@dataclass
class Trainer:
    datastore: Datastore
    data_module_class: Type[DataModule]
    batch_size: int
    time_step_count: int
    seed: int

    def objective(self, trial: optuna.trial.Trial) -> float:
        channel_count = self.data_module_class.channel_count
        class_count = self.data_module_class.class_count
        model = get_model(trial, channel_count, class_count)
        optimizer = get_optimizer(trial, model)

        generator = torch.Generator(device=model.device).manual_seed(self.seed)

        data_module = self.data_module_class(
            self.datastore,
            generator=generator,
            lengths=(0.8, 0.15, 0.05),
            image_size=model.sample_size,
            batch_size=self.batch_size,
        )
        train_dataloader = data_module.train_dataloader()
        learning_rate_scheduler = get_learning_rate_scheduler(
            optimizer, train_dataloader
        )
        val_dataloader = data_module.val_dataloader()
        return train(
            trial,
            generator,
            model,
            optimizer,
            learning_rate_scheduler,
            train_dataloader,
            val_dataloader,
            epoch_count,
            self.time_step_count,
        )

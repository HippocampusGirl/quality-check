from pprint import pformat
from typing import Iterator

import numpy as np
import optuna
import scipy
import torch
from accelerate import Accelerator
from diffusers import PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from torch.nn.functional import mse_loss
from tqdm.auto import tqdm

from ..logging import logger
from ..utils import TrainingState


def validate(
    trial: optuna.trial.Trial,
    state: TrainingState,
    accelerator: Accelerator,
    autoencoder_model: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    lpips: torch.nn.Module,
    val_iterator: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_count: int,
    val_timestep_count: int,
    val_timestep_pruning: int,
) -> float:
    autoencoder_model = autoencoder_model.eval()
    diffusion_model = diffusion_model.eval()

    with torch.no_grad():
        noise_scheduler = PNDMScheduler()
        noise_scheduler.set_timesteps(num_inference_steps=val_timestep_count)  # type: ignore
        timesteps = noise_scheduler.timesteps.to(accelerator.device)  # type: ignore

        good_bad_labels = torch.zeros(0, dtype=torch.int64)
        loss_values = torch.zeros(0, dtype=torch.float32)

        for clean_images, class_labels, _good_bad_labels in tqdm(
            val_iterator,
            leave=False,
            unit="batches" + " ",
            position=2,
            disable=not accelerator.is_main_process,
        ):
            _loss_values = torch.zeros_like(_good_bad_labels, dtype=torch.float32)

            clean_images = VaeImageProcessor.normalize(clean_images)
            clean_latents = (
                autoencoder_model.encode(clean_images).latents
                * autoencoder_model.config.scaling_factor
            )

            for start in tqdm(
                timesteps[1::val_timestep_pruning], leave=False, position=3
            ):
                # Sample noise to add to the images
                noise = torch.randn_like(clean_latents)

                noisy_latents = noise_scheduler.add_noise(clean_latents, noise, start)  # type: ignore

                for step in tqdm(
                    timesteps[timesteps <= start], leave=False, position=4
                ):
                    (predicted_noise,) = diffusion_model(
                        noisy_latents,
                        step,
                        class_labels=class_labels,
                        return_dict=False,
                    )
                    (noisy_latents,) = noise_scheduler.step(  # type: ignore
                        predicted_noise.clone(), step, noisy_latents, return_dict=False
                    )

                reconstructed_images = autoencoder_model.decode(
                    noisy_latents / autoencoder_model.config.scaling_factor
                ).sample.clamp(min=-1.0, max=1.0)
                _loss_values += mse_loss(
                    reconstructed_images, clean_images, reduction="none"
                ).mean(dim=(1, 2, 3))
                _loss_values += lpips(reconstructed_images, clean_images).squeeze()

            good_bad_labels = torch.cat(
                (good_bad_labels, _good_bad_labels.to(good_bad_labels.device)), dim=0
            )
            loss_values = torch.cat(
                (loss_values, _loss_values.to(good_bad_labels.device)), dim=0
            )

            good_count = (good_bad_labels == 0).sum().item()
            bad_count = (good_bad_labels == 1).sum().item()
            if min(good_count, bad_count) > val_count:  # Stop early
                break

    # T-test for difference by label
    pearsonr = scipy.stats.pearsonr(good_bad_labels.numpy(), loss_values.numpy())
    r = pearsonr.statistic
    if not np.isfinite(r):
        raise ValueError
    r2 = float(np.square(r).item())

    logs = dict(
        epoch=state.epoch_index.item(),
        val_loss=loss_values.mean().item(),
        r2=r2,
    )
    logger.info(f"Validation {pformat(logs)}")
    accelerator.log(logs, step=state.step_index.item())

    trial.report(r2, int(state.step_index.item()))
    if trial.should_prune():
        raise optuna.TrialPruned()

    return r2

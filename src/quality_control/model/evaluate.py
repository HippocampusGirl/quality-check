import numpy as np
import scipy
import torch
from diffusers import PNDMScheduler
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    step_count: int,
    step_pruning: int,
) -> float:
    noise_scheduler = PNDMScheduler()
    noise_scheduler.set_timesteps(num_inference_steps=step_count)  # type: ignore
    timesteps = noise_scheduler.timesteps.to(model.device)  # type: ignore

    good_bad_labels = torch.zeros(0, dtype=torch.int64)
    loss_values = torch.zeros(0, dtype=torch.float32)

    for clean_images, class_labels, _good_bad_labels in tqdm(
        val_dataloader, leave=False, position=0
    ):
        _loss_values = torch.zeros_like(_good_bad_labels, dtype=torch.float32)
        for start in tqdm(timesteps[1::step_pruning], leave=False, position=1):
            # Sample noise to add to the images
            noise = torch.randn_like(clean_images)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, start)  # type: ignore

            for step in tqdm(timesteps[timesteps <= start], leave=False, position=2):
                (predicted_noise,) = model(
                    noisy_images,
                    step,
                    class_labels=class_labels,
                    return_dict=False,
                )
                (noisy_images,) = noise_scheduler.step(  # type: ignore
                    predicted_noise.clone(), step, noisy_images, return_dict=False
                )

            denoised_images = (noisy_images / 2 + 0.5).clamp(0, 1)
            _loss_values += mse_loss(
                denoised_images, clean_images, reduction="none"
            ).mean(dim=(1, 2, 3))

        good_bad_labels = torch.cat((good_bad_labels, _good_bad_labels), dim=0)
        loss_values = torch.cat((loss_values, _loss_values), dim=0)

        good_count = (good_bad_labels == 0).sum().item()
        bad_count = (good_bad_labels == 1).sum().item()
        if min(good_count, bad_count) > 20:  # Stop early
            break

    # T-test for difference by label
    pearsonr = scipy.stats.pearsonr(
        good_bad_labels.cpu().numpy(), loss_values.cpu().numpy()
    )
    r = pearsonr.statistic
    if not np.isfinite(r):
        raise ValueError
    return float(np.square(r))

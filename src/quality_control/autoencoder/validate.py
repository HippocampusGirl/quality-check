from itertools import islice
from pprint import pformat
from typing import Iterator

import torch
from accelerate import Accelerator
from torch.nn.functional import mse_loss
from tqdm.auto import tqdm

from ..logging import logger
from ..utils import TrainingState


def get_channels(images: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            images[:, 0, torch.newaxis, ...].repeat(1, 3, 1, 1),
            images[:, 1, torch.newaxis, ...].repeat(1, 3, 1, 1),
        ),
        dim=2,
    )


def validate(
    state: TrainingState,
    accelerator: Accelerator,
    model: torch.nn.Module,
    val_iterator: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_count: int,
) -> None:
    model = accelerator.unwrap_model(model)
    model = model.eval()

    with torch.no_grad():
        numerator: float = 0.0

        for images, _, _ in tqdm(
            islice(val_iterator, val_count),
            total=val_count,
            leave=False,
            unit="batches" + " ",
            position=1,
            disable=not accelerator.is_main_process,
        ):
            batch_size = images.size(0)

            reconstructed_images, _ = model(images, return_dict=False)
            loss = mse_loss(reconstructed_images, images)

            average_loss: torch.Tensor = accelerator.gather(
                loss.repeat(batch_size)
            ).mean()

            numerator += average_loss.cpu().item()

    logs = dict(
        epoch=state.epoch_index,
        autoencoder_val_loss=numerator / val_count,
    )

    if accelerator.is_main_process:
        logger.info(f"Validation {pformat(logs)}")
        accelerator.log(logs, step=state.step_index)
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                channels = get_channels(images)
                reconstructed_channels = get_channels(reconstructed_images)

                tiled = torch.cat(
                    [channels, reconstructed_channels],
                    dim=3,
                )

                tracker.log_images(
                    dict(images=tiled),
                    step=state.step_index,
                    dataformats="NCHW",
                )

    torch.cuda.empty_cache()

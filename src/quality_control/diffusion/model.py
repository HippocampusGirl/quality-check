from pprint import pformat

import bitsandbytes
import optuna
import torch
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from ..logging import logger

block_types = ["{direction}Block2D", "Attn{direction}Block2D"]


def get_model(
    trial: optuna.trial.Trial,
    channel_count: int,
    class_count: int,
) -> torch.nn.Module:
    block_count = trial.suggest_int("block_count", 6, 8)
    down_block_types: list[str] = list()
    up_block_types: list[str] = list()
    for i in range(block_count):
        if i in [0, 1]:
            block_type = block_types[0]
        else:
            block_type = trial.suggest_categorical(f"block_type_{i}", block_types)
        down_block_types.append(block_type.format(direction="Down"))
        up_block_types.insert(0, block_type.format(direction="Up"))

    norm_group_count = 32
    block_out_channels_base = 32 * trial.suggest_int(
        "block_out_channels_factor", 32 // norm_group_count, 128 // norm_group_count
    )
    block_out_channels: list[int] = [block_out_channels_base]
    block_out_channels_factor = 1
    for i in range(1, block_count):
        block_out_channels_factor += trial.suggest_int(
            f"block_out_channels_factor_increment_{i}", 0, 1
        )
        block_out_channels.append(block_out_channels[0] * block_out_channels_factor)

    image_size_base = 2 ** (block_count - 1)
    image_size = image_size_base * trial.suggest_int(
        "image_size_factor", 256 // image_size_base, 768 // image_size_base
    )

    model_kwargs = dict(
        sample_size=image_size,
        in_channels=channel_count,
        out_channels=channel_count,
        center_input_sample=True,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=trial.suggest_int("layers_per_block", 1, 3),
        downsample_type="conv",
        upsample_type="conv",
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        norm_num_groups=norm_group_count,
        num_class_embeds=class_count,
    )

    logger.info(
        f"Creating model {pformat(model_kwargs)}",
    )

    model: torch.nn.Module = UNet2DModel(**model_kwargs)  # type: ignore

    return model


def get_optimizer(
    trial: optuna.trial.Trial, model: torch.nn.Module, suffix: str = ""
) -> torch.optim.Optimizer:
    name = "learning_rate"
    if suffix:
        name += f"_{suffix}"
    learning_rate = trial.suggest_float(name, 1e-5, 5e-2, log=True)
    optimizer = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=learning_rate)
    return optimizer  # type: ignore


def get_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epoch_count: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * epoch_count),
    )

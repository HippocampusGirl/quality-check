import torch
from diffusers import VQModel

from ..diffusion.model import get_learning_rate_scheduler as get_learning_rate_scheduler
from .discriminator import Discriminator

model_class = VQModel

image_size: int = 512


def get_model(channel_count: int) -> torch.nn.Module:
    # https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder/blob/main/movq/config.json
    model = model_class(
        in_channels=channel_count,
        out_channels=channel_count,
        act_fn="silu",
        block_out_channels=[128, 256, 256, 512],
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
        ],
        latent_channels=4,
        layers_per_block=2,
        norm_num_groups=32,
        norm_type="spatial",
        num_vq_embeddings=16384,
        sample_size=32,
        scaling_factor=0.18215,
        up_block_types=[
            "AttnUpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
    )
    return model  # type: ignore


def get_discriminator_model(channel_count: int) -> torch.nn.Module:
    return Discriminator(in_channels=channel_count)  # type: ignore

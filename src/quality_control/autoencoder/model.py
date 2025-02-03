import torch
from diffusers import VQModel

from ..diffusion.model import get_learning_rate_scheduler as get_learning_rate_scheduler
from .discriminator import Discriminator

model_class = VQModel
model_name = "f4g"

image_size: int = 256


def get_model(channel_count: int) -> torch.nn.Module:
    model = model_class(
        in_channels=channel_count,
        out_channels=channel_count,
        act_fn="silu",
        block_out_channels=[
            128,
            256,
            256,
            512,
        ],
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        latent_channels=4,
        vq_embed_dim=4,
        layers_per_block=2,
        norm_num_groups=32,
        norm_type="spatial",
        num_vq_embeddings=16384,
        sample_size=32,
        scaling_factor=0.18215,
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
    )
    model.quantize.legacy = False  # type: ignore
    return model  # type: ignore


def get_discriminator_model(channel_count: int) -> torch.nn.Module:
    return Discriminator(in_channels=channel_count)  # type: ignore

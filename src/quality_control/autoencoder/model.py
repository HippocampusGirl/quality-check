import torch
from diffusers import VQModel

from ..diffusion.model import get_learning_rate_scheduler as get_learning_rate_scheduler
from ..diffusion.model import get_optimizer as get_optimizer
from .discriminator import Discriminator

model_class = VQModel


def get_model(channel_count: int) -> torch.nn.Module:
    # https://github.com/huggingface/diffusers/blob/main/examples/vqgan/train_vqgan.py#L603
    # Taken from config of movq at kandinsky-community/kandinsky-2-2-decoder but without
    # the attention layers
    model = model_class(
        in_channels=channel_count,
        out_channels=channel_count,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        block_out_channels=[
            128,
            256,
            512,
        ],
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        sample_size=32,
        num_vq_embeddings=16384,
        norm_num_groups=32,
        vq_embed_dim=4,
        scaling_factor=0.18215,
        norm_type="spatial",
    )
    return model  # type: ignore


def get_discriminator_model(channel_count: int) -> torch.nn.Module:
    return Discriminator(in_channels=channel_count)  # type: ignore

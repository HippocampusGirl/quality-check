from functools import partial
from operator import methodcaller
from typing import Any

import torch
from lpips import LPIPS as _LPIPS
from torch.nn.functional import l1_loss, mse_loss

from ..utils import TrainingState
from .base import Model


def grad(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
    )[0].detach()


def is_compiled_module(module: Any) -> bool:
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def extract_model_from_parallel(model: Any) -> Any:
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    is_compiled = is_compiled_module(model)
    if is_compiled:
        compiled_model = model
        model = model._orig_mod

    while isinstance(model, options):
        model = model.module

    if is_compiled:
        compiled_model._orig_mod = model
        model = compiled_model

    return model


class LPIPS(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = _LPIPS(net="alex", verbose=False)

    def forward(self, *args: torch.Tensor) -> Any:
        batch_size, channel_count = args[0].size(0), args[0].size(1)
        loss = self.net(
            *(
                torch.flatten(arg, start_dim=0, end_dim=1)
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
                for arg in args
            ),
            normalize=False,
        )
        loss = loss.squeeze()
        loss = torch.unflatten(loss, 0, (batch_size, channel_count))
        loss = loss.mean(dim=1)
        return loss


class JukeboxLoss(torch.nn.Module):
    # Adapted from https://github.com/Project-MONAI/MONAI/blob/main/monai/losses/spectral_loss.py#L22
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        fft = map(partial(torch.fft.rfft2, norm="ortho"), args)
        amplitude = map(methodcaller("abs"), fft)

        # Compute distance between amplitude of frequency components
        # See Section 3.3 from https://arxiv.org/abs/2005.00341
        return mse_loss(*amplitude)


class AutoencoderLoss(torch.nn.Module):
    def __init__(
        self,
        state: TrainingState,
        autoencoder: Model,
        discriminator: Model,
        discriminator_warmup_steps: torch.Tensor,
        lpips: LPIPS,
    ):
        super().__init__()
        self.state = state

        self.autoencoder = autoencoder
        self.discriminator = discriminator

        self.discriminator_warmup_steps = discriminator_warmup_steps
        self.jukebox_loss = JukeboxLoss()
        self.lpips = lpips

    def get_adaptive_weight(
        self, loss: torch.Tensor, discriminator_loss: torch.Tensor
    ) -> Any:
        decoder_layer = extract_model_from_parallel(
            self.autoencoder.model
        ).decoder.conv_out.weight
        norm_grad_perceptual_loss = grad(loss, decoder_layer).norm(p=2)
        norm_grad_discriminator_loss = grad(discriminator_loss, decoder_layer).norm(p=2)
        adaptive_weight = (
            norm_grad_perceptual_loss / norm_grad_discriminator_loss.clamp(min=1e-4)
        ).clamp(min=0.0, max=1e4)

        return adaptive_weight

    def forward(
        self, images: torch.Tensor, reconstructed_images: torch.Tensor
    ) -> torch.Tensor:
        images = images.float()
        reconstructed_images = reconstructed_images.float()

        loss = l1_loss(reconstructed_images, images)

        perceptual_loss = self.lpips(reconstructed_images, images).mean()
        # https://github.com/marksgraham/ddpm-ood/blob/main/src/trainers/vqvae_trainer.py#L101
        loss += 0.001 * perceptual_loss

        loss += self.jukebox_loss(reconstructed_images, images)

        is_warmup = self.state.step_index < self.discriminator_warmup_steps
        if not is_warmup:
            discriminator_loss = -self.discriminator.model(reconstructed_images).mean()
            # https://github.com/marksgraham/ddpm-ood/blob/main/train_vqvae.py#L71
            loss += (
                self.get_adaptive_weight(loss, discriminator_loss) * discriminator_loss
            )

        return loss

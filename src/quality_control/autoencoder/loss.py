from dataclasses import dataclass
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from lpips import LPIPS as _LPIPS

from ..utils import TrainingState


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


def get_jukebox_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Adapted from https://github.com/Project-MONAI/MONAI/blob/main/monai/losses/spectral_loss.py#L22
    fft_a, fft_b = torch.fft.rfft2(a, norm="ortho"), torch.fft.rfft2(b, norm="ortho")
    amplitude_a, amplitude_b = fft_a.abs(), fft_b.abs()

    # Compute distance between amplitude of frequency components
    # See Section 3.3 from https://arxiv.org/abs/2005.00341
    return torch.nn.functional.mse_loss(amplitude_a, amplitude_b)


@dataclass
class GradNormOutput:
    reconstructed_images: torch.Tensor
    values: torch.Tensor
    weights: torch.Tensor
    autoencoder_loss: torch.Tensor
    commit_loss: torch.Tensor
    grad_norm_loss: torch.Tensor


class GradNorm(ModelMixin, ConfigMixin):  # type: ignore
    @register_to_config  # type: ignore
    def __init__(self, discriminator_warmup_steps: int, alpha: float = 1.5) -> None:
        super().__init__()  # type: ignore
        self.discriminator_warmup_steps = discriminator_warmup_steps
        self.alpha = alpha
        self.register_parameter("weights", torch.nn.Parameter(torch.ones(4)))

    def forward(
        self,
        state: TrainingState,
        autoencoder_model: torch.nn.Module,
        discriminator_model: torch.nn.Module,
        lpips: LPIPS,
        images: torch.Tensor,
    ) -> GradNormOutput:
        reconstructed_images, commit_loss = autoencoder_model(images, return_dict=False)
        if reconstructed_images is None:
            raise ValueError

        l1_loss = torch.nn.functional.l1_loss(reconstructed_images, images)
        lpips_loss = lpips(reconstructed_images, images).mean()
        jukebox_loss = get_jukebox_loss(reconstructed_images, images)
        adversarial_loss = torch.nn.functional.relu(
            1.0 - discriminator_model(reconstructed_images).mean()
        )

        values = torch.stack([l1_loss, lpips_loss, jukebox_loss, adversarial_loss])

        is_warmup = state.step_index < self.discriminator_warmup_steps
        size = torch.where(is_warmup, 3, 4)

        values = values[0:size]
        weights = self.weights[0:size]

        weights = torch.nn.functional.softplus(weights)
        weights = weights / weights.sum().clamp(min=1e-2) * torch.numel(weights)

        loss = (values * weights.detach()).mean() + commit_loss

        last_layer = extract_model_from_parallel(
            autoencoder_model
        ).decoder.conv_out.weight

        def get_grad_norm(value: torch.Tensor) -> Any:
            return torch.autograd.grad(
                outputs=value,
                inputs=last_layer,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )[0].norm(p=2)

        grad_norms = (
            torch.stack([get_grad_norm(value) for value in values]).detach() * weights
        )
        constant_term = grad_norms.mean() * values.detach().pow(self.alpha)

        grad_norm_loss = torch.nn.functional.l1_loss(grad_norms, constant_term)

        return GradNormOutput(
            reconstructed_images=reconstructed_images,
            values=values,
            weights=weights,
            autoencoder_loss=loss,
            commit_loss=commit_loss,
            grad_norm_loss=grad_norm_loss,
        )

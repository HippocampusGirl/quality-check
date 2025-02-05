from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


# Taken from https://github.com/huggingface/diffusers/blob/main/examples/vqgan/discriminator.py
# Discriminator model ported from Paella https://github.com/dome272/Paella/blob/main/src_distributed/vqgan.py
class Discriminator(ModelMixin, ConfigMixin):  # type: ignore
    @register_to_config  # type: ignore
    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 0,
        hidden_channels: int = 512,
        depth: int = 6,
    ) -> None:
        super().__init__()  # type: ignore
        d = max(depth - 3, 3)
        layers = [
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    in_channels,
                    hidden_channels // (2**d),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ),
            torch.nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            c_in = hidden_channels // (2 ** max((d - i), 0))
            c_out = hidden_channels // (2 ** max((d - 1 - i), 0))
            layers.append(
                torch.nn.utils.spectral_norm(
                    torch.nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
                )
            )
            layers.append(torch.nn.InstanceNorm2d(c_out))
            layers.append(torch.nn.LeakyReLU(0.2))
        self.encoder = torch.nn.Sequential(*layers)
        self.shuffle = torch.nn.Conv2d(
            (hidden_channels + cond_channels) if cond_channels > 0 else hidden_channels,
            1,
            kernel_size=1,
        )
        self.logits = torch.nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(
                cond.size(0),
                cond.size(1),
                1,
                1,
            ).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        x = self.logits(x)
        return x


def random_weighted_average(
    images: torch.Tensor, reconstructed_images: torch.Tensor
) -> Any:
    weights = torch.rand_like(images)
    return (weights * images) + ((1.0 - weights) * reconstructed_images)


def gradient_penalty(
    output: torch.Tensor, images: torch.Tensor, weight: float = 10.0
) -> torch.Tensor:
    (gradients,) = torch.autograd.grad(
        outputs=output,
        inputs=images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients.detach_()
    slopes = gradients.pow(2).sum(dim=(1, 2, 3)).sqrt()
    return weight * (slopes - 1).pow(2).mean()

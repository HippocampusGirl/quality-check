from typing import Any

import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import InterpolationMode


class RandomHorizontalResizedCrop(transforms.Transform):  # type: ignore
    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ):
        super().__init__()

        self.scale = scale

        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        height, width = transforms.query_size(flat_inputs)
        scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        w = int(round(width * scale))
        left = int(round(torch.empty(1).uniform_(0, width - w).item()))
        return dict(top=0, height=height, left=left, width=w)

    def _transform(self, inpt: Any, params: dict[str, float]) -> Any:
        return self._call_kernel(
            transforms.functional.resized_crop,
            inpt,
            **params,
            size=self.size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )

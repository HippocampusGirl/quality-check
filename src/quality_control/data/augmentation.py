from typing import Any

import cv2
from albumentations.augmentations.crops.transforms import _BaseRandomSizedCrop


class RandomHorizontalResizedCrop(_BaseRandomSizedCrop):  # type: ignore
    """
    RamdomA transformation
    """

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float],
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        p: float = 1.0,
    ) -> None:
        super().__init__(
            size=size,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            p=p,
        )
        self.scale = scale

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        shape = params["shape"][:2]
        height, width = shape

        scale = self.py_random.uniform(*self.scale)
        w = int(round(width * scale))
        left = int(round(self.py_random.uniform(0, width - w)))
        return dict(crop_coords=(left, 0, left + w, height))


# class RandomHorizontalResizedCrop(transforms.Transform):  # type: ignore
#     def __init__(
#         self,
#         size: tuple[int, int],
#         scale: tuple[float, float],
#         interpolation: InterpolationMode = InterpolationMode.BILINEAR,
#         antialias: bool = True,
#     ):
#         super().__init__()

#         self.scale = scale

#         self.size = size
#         self.interpolation = interpolation
#         self.antialias = antialias

#     def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
#         height, width = transforms.query_size(flat_inputs)
#         scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
#         w = int(round(width * scale))
#         left = int(round(torch.empty(1).uniform_(0, width - w).item()))
#         return dict(top=0, height=height, left=left, width=w)

#     def _transform(self, inpt: Any, params: dict[str, float]) -> Any:
#         return self._call_kernel(
#             transforms.functional.resized_crop,
#             inpt,
#             **params,
#             size=self.size,
#             interpolation=self.interpolation,
#             antialias=self.antialias,
#         )

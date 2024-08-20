from pathlib import Path
from typing import Iterator

import numpy as np
from numpy import typing as npt

from .segmentation import parse_segmentation


def parse_skull_strip(image_path: str | Path) -> Iterator[npt.NDArray[np.uint8]]:
    return parse_segmentation(image_path, ["#f77189"])

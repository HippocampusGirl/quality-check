from pathlib import Path
from typing import Iterator

from .base import Report
from .segmentation import parse_segmentation


def parse_skull_strip(image_path: str | Path) -> Iterator[Report]:
    return parse_segmentation(image_path, ["#f77189"])

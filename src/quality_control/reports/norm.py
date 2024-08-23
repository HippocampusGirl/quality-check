from pathlib import Path
from typing import Iterator

import scipy

from .base import Report
from .segmentation import parse_segmentation

colors = [
    "#f77189",  # 1 - Cortical ribbon
    "#dc8932",  # 2 - Lateral ventricle
    "#ae9d31",  # 3 - Thalamus
    "#77ab31",  # 4 - Caudate
    "#33b07a",  # 5 - Putamen
    "#36ada4",  # 6 - Globus pallidus external
    "#38a9c5",  # 7 - Globus pallidus internal
    "#6e9bf4",  # 8 - Hippocampus
    "#cc7af4",  # 9 - Amygdala
    "#f565cc",  # 10 - Cerebellum
    "red",  # Brain mask
]


def parse_norm(image_path: str | Path) -> Iterator[Report]:
    for report in parse_segmentation(image_path, colors):
        image = report.image

        # Fix overlapping masks (bug in HALFpipe)
        for i in range(1, image.shape[2]):
            masks = image[..., (i + 1) : -1]  # excluding brain mask

            overlap = masks.any(axis=-1)
            overlap = scipy.ndimage.binary_dilation(overlap, iterations=2)

            image[overlap, i] = 0

        yield report

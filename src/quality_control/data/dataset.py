from dataclasses import dataclass
from enum import IntEnum, auto
from operator import attrgetter

import numpy as np
from numpy import typing as npt
from torch.utils.data import Dataset as _Dataset

from .compression import decompress_image
from .schema import Datastore


class ImageType(IntEnum):
    skull_strip_report = auto()
    t1_norm_rpt = auto()
    tsnr_rpt = auto()
    bold_conf = auto()
    epi_norm_rpt = auto()


class ChannelType(IntEnum):
    t1w_scanner = 0
    brain_mask = 1

    t1w_standard = 2
    cortical_ribbon = 3
    lateral_ventricle = 4
    thalamus = 5
    caudate = 6
    putamen = 7
    gpe = 8
    gpi = 9
    hippocampus = 10
    amygdala = 11
    cerebellum = 12

    tsnr = 13
    carpet_plot = 14
    gs = 15
    gscsf = 16
    gswm = 17
    dvars = 18
    fd = 19

    epi_standard = 20


norm_overlays = (
    ChannelType.cortical_ribbon,
    ChannelType.lateral_ventricle,
    ChannelType.thalamus,
    ChannelType.caudate,
    ChannelType.putamen,
    ChannelType.gpe,
    ChannelType.gpi,
    ChannelType.hippocampus,
    ChannelType.amygdala,
    ChannelType.cerebellum,
    ChannelType.brain_mask,
)

channels_by_image_type: dict[ImageType, tuple[ChannelType, ...]] = {
    ImageType.skull_strip_report: (ChannelType.t1w_scanner, ChannelType.brain_mask),
    ImageType.t1_norm_rpt: (ChannelType.t1w_standard, *norm_overlays),
    ImageType.tsnr_rpt: (ChannelType.tsnr, ChannelType.brain_mask),
    ImageType.bold_conf: (
        ChannelType.carpet_plot,
        ChannelType.gs,
        ChannelType.gscsf,
        ChannelType.gswm,
        ChannelType.dvars,
        ChannelType.fd,
    ),
    ImageType.epi_norm_rpt: (ChannelType.epi_standard, *norm_overlays),
}
binary_channels = np.fromiter((n.value for n in norm_overlays), dtype=np.uint8)


@dataclass(frozen=True)
class Image:
    position: npt.NDArray[np.uint8]
    channels: npt.NDArray[np.uint8]
    data: npt.NDArray[np.uint8]


def get_channels(image_type: ImageType) -> npt.NDArray[np.uint8]:
    return np.fromiter(
        map(attrgetter("value"), channels_by_image_type[image_type]), dtype=np.uint8
    )


class ImageDataset(_Dataset[Image]):
    def __init__(self, datastore: Datastore, **query_dict: str):
        self.datastore = datastore
        self.images: list[tuple[ImageType, npt.NDArray[np.uint8], int]] = list()

        self.query = frozenset(query_dict.items())

        with self.datastore:
            image_ids_by_tags = datastore.get_image_ids_by_tags()
            for tags_set, image_ids in image_ids_by_tags.items():
                if not self.query.issubset(tags_set):
                    continue
                tags = dict(tags_set)
                image_type = ImageType[tags["suffix"]]
                for image_id in image_ids:
                    direction_str, index = datastore.get_direction_and_index(image_id)
                    position = np.zeros(3, dtype=np.uint8)
                    if direction_str is not None and index is not None:
                        if index == 0:
                            raise ValueError
                        position[["x", "y", "z"].index(direction_str)] = index
                    self.images.append((image_type, position, image_id))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Image:
        image_type, position, image_id = self.images[index]
        with self.datastore:
            image_bytes = self.datastore.get_image(image_id)
        channels = get_channels(image_type)

        data = decompress_image(image_bytes)
        is_binary = np.isin(channels, binary_channels)
        if not np.isin(data[..., is_binary], (0, 1)).all():
            raise ValueError
        data[..., is_binary] = np.where(data[..., is_binary], 255, 0)

        return Image(
            position=position,
            channels=channels,
            data=data,
        )

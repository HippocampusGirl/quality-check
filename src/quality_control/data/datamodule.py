from functools import partial
from itertools import chain
from typing import Any, ClassVar, Iterable, Sized, TypeAlias

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from numpy import typing as npt
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    WeightedRandomSampler,
    random_split,
)
from torch.utils.data import Dataset as _Dataset
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import InterpolationMode

from ..utils import cpu_count
from .dataset import (
    Image,
    ImageDataset,
    ImageType,
    channels_by_image_type,
    get_channels,
)
from .schema import Datastore


def get_weights(datasets: Iterable[Sized]) -> npt.NDArray[np.float32]:
    return np.fromiter(
        chain.from_iterable([1 / len(dataset)] * len(dataset) for dataset in datasets),
        dtype=np.float32,
    )


class BaseDataModule(LightningDataModule):
    classes: list[tuple[np.uint8, ...]]

    channel_count: ClassVar[int]
    class_count: ClassVar[int]

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        image_types: list[ImageType],
        preprocess: transforms.Transform,
        lengths: tuple[float, float, float],
        train_batch_size: int,
        eval_batch_size: int,
    ):
        super().__init__()

        self.generator = generator

        if self.generator.device.type == "cpu":
            self.num_workers = cpu_count()
        else:
            self.num_workers = 0

        if generator.device.type == "cuda":
            preprocess = torch.compile(preprocess)
        self.preprocess: Any = preprocess

        self.lengths = lengths

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        if {self.channel_count} != {
            len(channels_by_image_type[image_type]) for image_type in image_types
        }:
            raise ValueError

        good_datasets = {
            image_type: ImageDataset(datastore, label="good", suffix=image_type.name)
            for image_type in image_types
        }
        self.classes = sorted(
            {
                (*tuple(get_channels(image_type)), *tuple(position))
                for image_type, dataset in good_datasets.items()
                for _, position, _ in dataset.images
            }
        )
        self.good_weights = get_weights(good_datasets.values())

        good_dataset: _Dataset[Image] = ConcatDataset(good_datasets.values())
        tensor_dataset_factory = partial(TensorDataset, self)
        self.train_dataset, val_dataset_good, test_dataset_good = map(
            tensor_dataset_factory,
            random_split(good_dataset, self.lengths, generator=self.generator),
        )

        bad_datasets = {
            image_type: ImageDataset(datastore, label="bad", suffix=image_type.name)
            for image_type in image_types
        }
        self.bad_weights = get_weights(bad_datasets.values())

        bad_dataset: _Dataset[Image] = ConcatDataset(bad_datasets.values())
        bad_lengths = tuple(
            length / sum(self.lengths[1:]) for length in self.lengths[1:]
        )
        val_dataset_bad, test_dataset_bad = map(
            tensor_dataset_factory,
            random_split(bad_dataset, bad_lengths, generator=self.generator),
        )
        self.val_dataset = GoodBadDataset(val_dataset_good, val_dataset_bad)
        self.test_dataset = GoodBadDataset(test_dataset_good, test_dataset_bad)

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return dict(
            drop_last=True,
            generator=self.generator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=False,
        )

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        dataset = self.train_dataset
        weights = self.good_weights[dataset.subset.indices]
        sampler = WeightedRandomSampler(
            list(weights), len(dataset), replacement=True, generator=self.generator
        )
        return DataLoader(
            dataset, self.train_batch_size, sampler=sampler, **self.dataloader_kwargs
        )

    def get_data_loader(
        self, dataset: "GoodBadDataset"
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        good_dataset, bad_dataset = dataset.good_dataset, dataset.bad_dataset
        good_weights = self.good_weights[good_dataset.subset.indices]
        good_weights /= good_weights.sum()
        bad_weights = self.bad_weights[bad_dataset.subset.indices]
        bad_weights /= bad_weights.sum()
        weights = np.hstack([good_weights, bad_weights])
        sampler = WeightedRandomSampler(
            list(weights), len(dataset), replacement=True, generator=self.generator
        )
        return DataLoader(
            dataset, self.eval_batch_size, sampler=sampler, **self.dataloader_kwargs
        )

    def val_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.get_data_loader(self.val_dataset)

    def test_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.get_data_loader(self.test_dataset)


class TensorDataset(_Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_module: BaseDataModule, subset: Subset[Image]):
        self.subset = subset
        self.data_module = data_module

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.subset[index]
        class_label = self.data_module.classes.index(
            (*tuple(image.channels), *tuple(image.position))
        )
        tensor = torch.tensor(image.data).permute(2, 0, 1)
        data = self.data_module.preprocess(tensor)
        return data, torch.tensor(class_label)


# Maybe use StackDataset instead of GoodBadDataset
class GoodBadDataset(_Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, good_dataset: TensorDataset, bad_dataset: TensorDataset):
        self.good_dataset = good_dataset
        self.bad_dataset = bad_dataset

    def __len__(self) -> int:
        return len(self.good_dataset) + len(self.bad_dataset)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < len(self.good_dataset):
            dataset = self.good_dataset
            index = index
            label = torch.tensor(0)
        else:
            dataset = self.bad_dataset
            index = index - len(self.good_dataset)
            label = torch.tensor(1)

        return *dataset[index], label


class SliceDataModule(BaseDataModule):
    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        image_types: list[ImageType],
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
    ):
        preprocess = transforms.Compose(
            [
                transforms.Resize(None, max_size=image_size),
                transforms.CenterCrop(size=(image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ElasticTransform(alpha=100.0, sigma=10.0),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        super().__init__(
            datastore,
            generator,
            image_types,
            preprocess,
            lengths,
            train_batch_size,
            eval_batch_size,
        )


class DataModule1(SliceDataModule):
    channel_count = 2
    class_count = 42

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
    ):
        image_types = [ImageType.skull_strip_report, ImageType.tsnr_rpt]
        super().__init__(
            datastore,
            generator,
            image_types,
            lengths,
            image_size,
            train_batch_size,
            eval_batch_size,
        )


class DataModule2(SliceDataModule):
    channel_count = 12
    class_count = 42

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
    ):
        image_types = [ImageType.t1_norm_rpt, ImageType.epi_norm_rpt]
        super().__init__(
            datastore,
            generator,
            image_types,
            lengths,
            image_size,
            train_batch_size,
            eval_batch_size,
        )


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


class DataModule3(BaseDataModule):
    channel_count = 6
    class_count = 1

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
    ):
        preprocess = transforms.Compose(
            [
                RandomHorizontalResizedCrop(
                    size=(image_size, image_size), scale=(0.3, 1.0)
                ),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        image_types = [ImageType.bold_conf]
        super().__init__(
            datastore,
            generator,
            image_types,
            preprocess,
            lengths,
            train_batch_size,
            eval_batch_size,
        )


DataModule: TypeAlias = DataModule1 | DataModule2 | DataModule3

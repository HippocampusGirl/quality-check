import random
from functools import partial
from typing import Any, Callable, ClassVar, Sequence, TypeAlias

import albumentations
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from diffusers.image_processor import VaeImageProcessor
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

from ..utils import apply_num_threads, cpu_count
from .augmentation import RandomHorizontalResizedCrop
from .dataset import (
    Image,
    ImageDataset,
    ImageType,
    get_channels,
)
from .schema import Datastore


def worker_init(worker_id: int) -> None:
    apply_num_threads(1)
    cv2.setNumThreads(1)


def get_weights(datasets: Sequence[ImageDataset]) -> npt.NDArray[np.float32]:
    weights: list[float] = list()

    average_dataset_size = sum(len(dataset) for dataset in datasets) / len(datasets)
    for dataset in datasets:
        dataset_size = len(dataset)
        if dataset_size == 0:
            continue

        sub = [sub for _, sub, _, _, _ in dataset.images]
        _, inverse, counts = np.unique(sub, return_inverse=True, return_counts=True)
        sub_count = counts.size
        sub_counts = counts[inverse]
        ds = [ds for ds, _, _, _, _ in dataset.images]
        _, inverse, counts = np.unique(ds, return_inverse=True, return_counts=True)
        ds_count = counts.size
        ds_counts = counts[inverse]

        average_sub_size = dataset_size / sub_count
        average_ds_size = dataset_size / ds_count
        w = (
            (average_sub_size / sub_counts)
            * (average_ds_size / ds_counts)
            * (average_dataset_size / len(dataset))
        )
        weights.extend(w)

    return np.array(weights)


class BaseDataModule(LightningDataModule):
    classes: list[tuple[np.uint8, ...]]

    channel_count: ClassVar[int]
    class_count: ClassVar[int]

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        image_types: list[ImageType],
        get_preprocess: Callable[..., Any],
        lengths: tuple[float, float, float],
        train_batch_size: int,
        eval_batch_size: int,
        autoencoder_model: torch.nn.Module | None = None,
    ):
        super().__init__()

        self.generator = generator
        random.seed(torch.randint(1 << 32, (), generator=generator).item())
        np.random.seed(torch.randint(1 << 32, (), generator=generator).item())

        if self.generator.device.type == "cpu":
            self.num_workers = cpu_count()
        else:
            self.num_workers = 0

        self.lengths = lengths

        train_preprocess = get_preprocess(is_augmentation=True)
        val_preprocess = get_preprocess(is_augmentation=False)

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        good_datasets = {
            image_type: ImageDataset(datastore, label="good", suffix=image_type.name)
            for image_type in image_types
        }
        self.classes = sorted(
            {
                (*tuple(get_channels(image_type)), *tuple(position))
                for image_type, dataset in good_datasets.items()
                for _, _, _, position, _ in dataset.images
            }
        )
        self.good_weights = get_weights(list(good_datasets.values()))

        good_dataset: _Dataset[Image] = ConcatDataset(good_datasets.values())
        tensor_dataset_factory = partial(TensorDataset, self.classes)
        _train_dataset, _val_dataset_good, _test_dataset_good = random_split(
            good_dataset, self.lengths, generator=self.generator
        )
        self.train_dataset = tensor_dataset_factory(
            train_preprocess, _train_dataset, autoencoder_model=autoencoder_model
        )
        val_dataset_factory = partial(tensor_dataset_factory, val_preprocess)
        val_dataset_good = val_dataset_factory(_val_dataset_good)
        test_dataset_good = val_dataset_factory(_test_dataset_good)

        bad_datasets = {
            image_type: ImageDataset(datastore, label="bad", suffix=image_type.name)
            for image_type in image_types
        }
        self.bad_weights = get_weights(list(bad_datasets.values()))

        bad_dataset: _Dataset[Image] = ConcatDataset(bad_datasets.values())
        bad_lengths = tuple(
            length / sum(self.lengths[1:]) for length in self.lengths[1:]
        )
        val_dataset_bad, test_dataset_bad = map(
            val_dataset_factory,
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
            worker_init_fn=worker_init,
            prefetch_factor=10,
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


@torch.compile()
def encode(autoencoder_model: torch.nn.Module, data: torch.Tensor) -> Any:
    data = VaeImageProcessor.normalize(data)
    return (
        autoencoder_model.encode(data).latents * autoencoder_model.config.scaling_factor
    )


class TensorDataset(_Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        classes: list[tuple[np.uint8, ...]],
        preprocess: Any,
        subset: Subset[Image],
        autoencoder_model: torch.nn.Module | None = None,
    ):
        self.preprocess = preprocess
        self.autoencoder_model: Any = autoencoder_model
        if self.autoencoder_model is not None:
            self.autoencoder_model = self.autoencoder_model.eval()

        self.classes = classes
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.subset[index]
        class_label = self.classes.index(
            (*tuple(image.channels), *tuple(image.position))
        )
        preprocessed_image: npt.NDArray[np.uint8] = self.preprocess(
            image=image.data.transpose(2, 0, 1)
        )
        data = transforms.functional.to_dtype(
            torch.tensor(preprocessed_image), scale=True
        )
        if self.autoencoder_model is not None:
            data = encode(self.autoencoder_model, data)
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
    @classmethod
    def get_preprocess(cls, image_size: int, is_augmentation: bool) -> Any:
        transform_list: list[albumentations.BasicTransform] = [
            albumentations.LongestMaxSize(
                max_size=image_size, interpolation=cv2.INTER_LINEAR
            ),
            albumentations.CenterCrop(
                height=image_size, width=image_size, pad_if_needed=True
            ),
        ]
        if is_augmentation:
            # scale = 0.05
            # angle = 5
            transform_list.extend(
                (
                    albumentations.HorizontalFlip(p=0.5),
                    # albumentations.Affine(
                    #     scale=(1 - scale, 1 + scale),
                    #     rotate=(-angle, angle),
                    #     translate_percent=(-0.05, 0.05),
                    #     shear=dict(x=(-angle, angle), y=(-angle, angle)),
                    #     interpolation=cv2.INTER_LINEAR,
                    #     p=1.0,
                    # ),
                    albumentations.GridDistortion(
                        num_steps=25, interpolation=cv2.INTER_LINEAR, p=1.0
                    ),
                    albumentations.RandomGamma(p=1.0),
                )
            )
        return albumentations.Compose(
            transform_list,
            additional_targets=dict(
                image0="image",
                mask0="mask",
                mask1="mask",
                mask2="mask",
                mask3="mask",
                mask4="mask",
                mask5="mask",
                mask6="mask",
                mask7="mask",
                mask8="mask",
                mask9="mask",
                mask10="mask",
            ),
        )

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        image_types: list[ImageType],
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
        autoencoder_model: torch.nn.Module | None = None,
    ):
        super().__init__(
            datastore,
            generator,
            image_types,
            partial(self.get_preprocess, image_size),
            lengths,
            train_batch_size,
            eval_batch_size,
            autoencoder_model=autoencoder_model,
        )


class DataModule1(SliceDataModule):
    channel_count = 2
    class_count = 42
    name = "m1"

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
        autoencoder_model: torch.nn.Module | None = None,
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
            autoencoder_model=autoencoder_model,
        )


class DataModule2(SliceDataModule):
    channel_count = 12
    class_count = 42
    name = "m2"

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
        autoencoder_model: torch.nn.Module | None = None,
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
            autoencoder_model=autoencoder_model,
        )


class DataModule3(BaseDataModule):
    channel_count = 6
    class_count = 1
    name = "m3"

    @classmethod
    def get_preprocess(cls, image_size: int, is_augmentation: bool) -> Any:
        transform_list: list[albumentations.BasicTransform] = list()
        if is_augmentation:
            transform_list.append(
                RandomHorizontalResizedCrop(
                    size=(image_size, image_size), scale=(0.3, 1.0), p=1.0
                )
            )
        else:
            transform_list.append(
                albumentations.Resize(height=image_size, width=image_size, p=1.0)
            )
        return albumentations.Compose(
            transform_list,
            additional_targets=dict(image0="image"),
        )

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
        autoencoder_model: torch.nn.Module | None = None,
    ):
        image_types = [ImageType.bold_conf]
        super().__init__(
            datastore,
            generator,
            image_types,
            partial(self.get_preprocess, image_size),
            lengths,
            train_batch_size,
            eval_batch_size,
            autoencoder_model=autoencoder_model,
        )


DataModule: TypeAlias = DataModule1 | DataModule2 | DataModule3


class TwoChannelDataModule(BaseDataModule):
    channel_count = 2
    class_count = 21 + 21 + 21 + 1 + 21
    name = "2c"

    def __init__(
        self,
        datastore: Datastore,
        generator: torch.Generator,
        lengths: tuple[float, float, float],
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
        autoencoder_model: torch.nn.Module | None = None,
    ):
        image_types = [
            ImageType.skull_strip_report,
            ImageType.t1_norm_rpt,
            ImageType.tsnr_rpt,
            ImageType.bold_conf,
            ImageType.epi_norm_rpt,
        ]
        super().__init__(
            datastore,
            generator,
            image_types,
            partial(self.get_preprocess, image_size),
            lengths,
            train_batch_size,
            eval_batch_size,
            autoencoder_model=autoencoder_model,
        )

    def get_preprocess(self, image_size: int, is_augmentation: bool) -> Any:
        preprocess_slice = SliceDataModule.get_preprocess(image_size, is_augmentation)
        preprocess_bold_conf = DataModule3.get_preprocess(image_size, is_augmentation)

        def preprocess(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
            channel_count = image.shape[
                -3
            ]  # shape is ..., channel_count, height, width
            if channel_count == 2:  # skull_strip_report and tsnr_rpt
                data_dict = preprocess_slice(image=image[0, ...], mask=image[1, ...])
                data_sequence = [data_dict["image"], data_dict["mask"]]
            elif channel_count == 6:
                data_dict = preprocess_bold_conf(
                    image=image[0, ...], image0=image[-1, ...]
                )
                data_sequence = [data_dict["image"], data_dict["image0"]]
            elif channel_count == 12:
                weights = np.array(
                    [170, 212, 43, 128, 149, 234, 234, 191, 128, 212, 255],
                    dtype=np.uint8,
                )
                keys = ["image", "mask", *(f"mask{i}" for i in range(10))]
                data_dict = preprocess_slice(**dict(zip(keys, image, strict=True)))
                data_sequence = [data_dict[key] for key in keys]
                data = np.stack(data_sequence)
                merge = (data[1:-1] * weights[:-1, np.newaxis, np.newaxis]).max(axis=0)
                merge[np.logical_and(merge == 0, data[-1] != 0)] = weights[-1]
                data_sequence = [data[0], merge]
            else:
                raise ValueError

            return np.stack(data_sequence)

        return preprocess

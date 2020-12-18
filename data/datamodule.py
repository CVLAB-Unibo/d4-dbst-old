from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np  # type: ignore
import torch
from hesiod import hcfg
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from data.dataset import Dataset
from data.transforms import TRANSFORM_T, Compose, Normalize, Resize
from data.utils import denormalize

ITEM_T = Tuple[Tensor, Optional[Tensor], Optional[Tensor]]


class DataModule(LightningDataModule):
    def __init__(self) -> None:
        LightningDataModule.__init__(self)  # type: ignore
        num_workers = hcfg("num_workers", int)
        self.build_dataloader = partial(
            DataLoader,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=Dataset.collate_fn,
        )

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        LightningDataModule.prepare_data(self, *args, **kwargs)  # type: ignore

    def setup(self, stage: Optional[str] = None) -> None:
        LightningDataModule.setup(self, stage)

        transforms: List[TRANSFORM_T] = []

        dummy = (1, 1)
        resize_dim = hcfg("resize_dim", type(dummy))
        transforms.append(Resize(resize_dim, resize_dim, resize_dim))

        dummy = (0.1, 0.1, 0.1)
        self.mean = hcfg("mean", type(dummy))
        self.std = hcfg("std", type(dummy))
        transforms.append(Normalize(self.mean, self.std))

        transform = Compose(transforms)

        self.train_dst = Dataset("train_dataset", transform)
        self.val_source_dst = Dataset("val_source_dataset", transform)
        self.val_target_dst = Dataset("val_target_dataset", transform)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        bs = hcfg("train_batch_size", int)
        return self.build_dataloader(self.train_dst, batch_size=bs, shuffle=True)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        bs = hcfg("val_batch_size", int)
        val_source_dl = self.build_dataloader(self.val_source_dst, batch_size=bs, shuffle=True)
        val_target_dl = self.build_dataloader(self.val_target_dst, batch_size=bs, shuffle=True)
        return [val_source_dl, val_target_dl]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        bs = hcfg("val_batch_size", int)
        test_source_dl = self.build_dataloader(self.val_source_dst, batch_size=bs, shuffle=True)
        test_target_dl = self.build_dataloader(self.val_target_dst, batch_size=bs, shuffle=True)
        return [test_source_dl, test_target_dl]

    def __len__(self) -> int:
        return len(self.train_dataloader())

    def transfer_batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        return batch

    @staticmethod
    def get_samples(dataloader: DataLoader, num_samples: int) -> ITEM_T:
        imgs: List[Tensor] = []
        sems: List[Tensor] = []
        deps: List[Tensor] = []

        iterator = iter(dataloader)
        for i in range(num_samples):
            img, sem, dep = next(iterator)
            imgs.append(img)
            sems.append(sem)
            deps.append(dep)

        imgt = torch.cat(imgs)
        semt = torch.cat(sems) if sems[0] is not None else None
        dept = torch.cat(deps) if deps[0] is not None else None

        return imgt, semt, dept

    def get_train_samples(self, num_samples: int) -> ITEM_T:
        return DataModule.get_samples(self.train_dataloader(), num_samples)

    def get_val_source_samples(self, num_samples: int) -> ITEM_T:
        dataloaders = cast(List, self.val_dataloader())
        return DataModule.get_samples(dataloaders[0], num_samples)

    def get_val_target_samples(self, num_samples: int) -> ITEM_T:
        dataloaders = cast(List, self.val_dataloader())
        return DataModule.get_samples(dataloaders[1], num_samples)

    def get_images(self, item: ITEM_T, dataset: Dataset) -> Sequence[Optional[np.ndarray]]:
        img, sem, dep = [element[0] if element is not None else None for element in item]

        denorm_img = np.transpose(np.array(img), axes=(1, 2, 0))
        denorm_img = denormalize(denorm_img, self.mean, self.std)

        colored_sem = dataset.sem_cmap(np.array(sem)) if sem is not None else None

        colored_dep = None
        if dep is not None:
            inv_dep = 1 / dep
            norm_dep = (inv_dep - inv_dep.min()) / (inv_dep.max() - inv_dep.min())
            colored_dep = dataset.dep_cmap(np.array(norm_dep))

        return denorm_img, colored_sem, colored_dep

    def get_train_images(self) -> Sequence[Optional[np.ndarray]]:
        item = self.get_train_samples(1)
        return self.get_images(item, self.train_dst)

    def get_val_source_images(self) -> Sequence[Optional[np.ndarray]]:
        item = self.get_val_source_samples(1)
        return self.get_images(item, self.val_source_dst)

    def get_val_target_images(self) -> Sequence[Optional[np.ndarray]]:
        item = self.get_val_target_samples(1)
        return self.get_images(item, self.val_target_dst)

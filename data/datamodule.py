from functools import partial
from typing import Any, List, Union

import torch
from hesiod import hcfg
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from data import ITEM_T
from data.dataset import Dataset
from data.transforms import Compose

num_workers = hcfg("num_workers", int)
build_dataloader = partial(DataLoader, num_workers=num_workers, pin_memory=True)


class DataModule(LightningDataModule):
    def __init__(self) -> None:
        LightningDataModule.__init__(self)  # type: ignore

        transform = Compose([])

        self.train_dst = Dataset(transform)
        self.val_source_dst = Dataset(transform)
        self.val_target_dst = Dataset(transform)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        bs = hcfg("train_batch_size", int)
        return build_dataloader(self.train_dst, batch_size=bs, shuffle=True)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        bs = hcfg("val_batch_size", int)
        val_source_dl = build_dataloader(self.val_source_dst, batch_size=bs, shuffle=True)
        val_target_dl = build_dataloader(self.val_target_dst, batch_size=bs, shuffle=True)
        return [val_source_dl, val_target_dl]

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
        dataloader = self.val_dataloader()
        if isinstance(dataloader, list):
            dataloader = dataloader[0]
        return DataModule.get_samples(dataloader, num_samples)

    def get_val_target_samples(self, num_samples: int) -> ITEM_T:
        dataloader = self.val_dataloader()
        if isinstance(dataloader, list):
            dataloader = dataloader[1]
        return DataModule.get_samples(dataloader, num_samples)

    def __len__(self) -> int:
        return len(self.train_dataloader())

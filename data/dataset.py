from abc import ABC
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import torch
from hesiod import hcfg
from PIL import Image  # type: ignore
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset as TorchDataset

from data.colormap import get_cmap
from data.semmap import get_map
from data.transforms import Compose
from data.utils import img2depth

SAMPLE_T = Sequence[Path]
ITEM_T = Tuple[Tensor, Optional[Tensor], Optional[Tensor]]


class Dataset(TorchDataset, ABC):
    def __init__(self, cfgkey: str, transform: Compose) -> None:
        TorchDataset.__init__(self)
        ABC.__init__(self)

        root = Path(hcfg("data_root", str))
        input_file = hcfg(f"{cfgkey}.input_file", str)
        self.samples: List[SAMPLE_T] = []

        with open(input_file, "rt") as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                splits = line.split(";")
                image = root / Path(splits[0].strip())
                sem = root / Path(splits[1].strip())
                dep = root / Path(splits[2].strip())
                self.samples.append((image, sem, dep))

        self.sem = hcfg(f"{cfgkey}.sem", bool)
        if self.sem:
            self.sem_map = get_map(hcfg(f"{cfgkey}.sem_map", str))
            self.sem_cmap = get_cmap(hcfg(f"{cfgkey}.sem_cmap", str))

        self.dep = hcfg(f"{cfgkey}.dep", bool)
        if self.dep:
            self.dep_min, self.dep_max = hcfg("dep_range", type((0.1, 0.1)))
            self.dep_cmap = get_cmap(hcfg(f"{cfgkey}.dep_cmap", str))

        self.transform = transform

    def encode_sem(self, sem_img: Image.Image) -> np.ndarray:
        sem = np.array(sem_img)
        sem_copy = sem.copy()
        for k, v in self.sem_map.items():
            sem_copy[sem == k] = v
        return sem_copy

    def __getitem__(self, index: int) -> ITEM_T:
        image_path, sem_path, dep_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        sem, dep = None, None
        if self.sem:
            sem = self.encode_sem(Image.open(sem_path))
        if self.dep:
            dep = img2depth(Image.open(dep_path))
            dep = np.clip(dep, self.dep_min, self.dep_max)

        sample = (image, sem, dep)
        return self.transform(sample)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def collate_fn(batches: List[ITEM_T]) -> ITEM_T:
        imgs: List[Tensor] = []
        sems: List[Tensor] = []
        deps: List[Tensor] = []

        for batch in batches:
            img, sem, dep = batch
            imgs.append(img)
            if sem is not None:
                sems.append(sem)
            if dep is not None:
                deps.append(dep)

        img_stack = torch.stack(imgs, 0)
        sem_stack = torch.stack(sems, 0) if len(sems) > 0 else None
        dep_stack = torch.stack(deps, 0) if len(deps) > 0 else None

        return img_stack, sem_stack, dep_stack

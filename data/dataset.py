from abc import ABC
from pathlib import Path
from typing import List, Sequence

import numpy as np  # type: ignore
from hesiod import hcfg
from PIL import Image  # type: ignore
from torch.utils.data.dataset import Dataset as TorchDataset

from data import ITEM_T
from data.colormap import get_cmap
from data.semmap import get_map
from data.transforms import Compose
from data.utils import img2depth

SAMPLE_T = Sequence[Path]


class Dataset(TorchDataset, ABC):
    def __init__(self, cfgkey: str, transform: Compose) -> None:
        TorchDataset.__init__(self)
        ABC.__init__(self)

        root = Path(hcfg(f"{cfgkey}.root", str))
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
            self.semmap = get_map(hcfg(f"{cfgkey}.semmap", str))
            self.semcmap = get_cmap(hcfg(f"{cfgkey}.semcmap", str))

        self.dep = hcfg(f"{cfgkey}.dep", bool)
        if self.dep:
            self.depcmap = get_cmap(hcfg(f"{cfgkey}.depcmap", str))

        self.mean = hcfg(f"{cfgkey}.mean", Sequence[float])
        self.std = hcfg(f"{cfgkey}.std", Sequence[float])
        self.transform = transform

    def encode_sem(self, sem_img: Image.Image) -> np.ndarray:
        sem = np.array(sem_img)
        sem_copy = sem.copy()
        for k, v in self.semmap.items():
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

        sample = (image, sem, dep)
        return self.transform(sample)

    def __len__(self) -> int:
        return len(self.samples)

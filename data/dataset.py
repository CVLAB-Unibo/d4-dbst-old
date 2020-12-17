from abc import ABC
from pathlib import Path
from typing import List, Tuple

import numpy as np  # type: ignore
from hesiod import hcfg
from PIL import Image  # type: ignore
from torch.utils.data.dataset import Dataset as TorchDataset

from data import ITEM_T
from data.semmap import get_map
from data.transforms import Compose
from data.utils import img2depth

SAMPLE_T = Tuple[Path, Path, Path]


class Dataset(TorchDataset, ABC):
    def __init__(self, transform: Compose) -> None:
        TorchDataset.__init__(self)
        ABC.__init__(self)

        root = Path(hcfg("dataset.root", str))
        input_file = hcfg("dataset.input_file", str)
        self.samples: List[SAMPLE_T] = []

        with open(input_file, "rt") as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                splits = line.split(";")
                image = root / Path(splits[0].strip())
                sem = root / Path(splits[1].strip())
                dep = root / Path(splits[2].strip())
                self.samples.append((image, sem, dep))

        self.dep = hcfg("dataset.dep", bool)
        self.sem = hcfg("dataset.sem", bool)
        if self.sem:
            self.semmap = get_map(hcfg("dataset.semmap", str))

        self.mean = hcfg("dataset.mean", Tuple[float])
        self.std = hcfg("dataset.std", Tuple[float])
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

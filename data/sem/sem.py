from abc import ABC

from data.dataset import Dataset


class SemDataset(Dataset, ABC):
    def __init__(self) -> None:
        Dataset.__init__(self)
        ABC.__init__(self)

from torch.utils.data.dataset import Dataset as TorchDataset
from abc import ABC


class Dataset(TorchDataset, ABC):
    def __init__(self) -> None:
        TorchDataset.__init__(self)
        ABC.__init__(self)

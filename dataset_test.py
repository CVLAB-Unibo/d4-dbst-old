from typing import List

import matplotlib.pyplot as plt  # type: ignore
from hesiod import hmain
from hesiod.core import hcfg

from data import DataModule
from data.transforms import TRANSFORM_T, Compose, Normalize, Resize


@hmain(base_cfg_dir="cfg", template_cfg_file="cfg/dataset_test.yaml")
def dataset_test() -> None:
    transforms: List[TRANSFORM_T] = []

    resize_dim = hcfg("resize_dim", type((1, 1)))
    transforms.append(Resize(resize_dim, resize_dim, resize_dim))

    mean = hcfg("mean", type((0.1, 0.1, 0.1)))
    std = hcfg("std", type((0.1, 0.1, 0.1)))
    transforms.append(Normalize(mean, std))

    transform = Compose(transforms)

    data_module = DataModule(train_transform=transform, val_transform=transform)
    data_module.prepare_data()
    data_module.setup()

    img, sem, dep = data_module.get_train_images()

    _, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(img)
    if sem is not None:
        axs[1].imshow(sem)
    if dep is not None:
        axs[2].imshow(dep)
    plt.show()


if __name__ == "__main__":
    dataset_test()

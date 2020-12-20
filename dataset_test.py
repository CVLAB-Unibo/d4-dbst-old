import matplotlib.pyplot as plt  # type: ignore
from hesiod import hmain

from data import DataModule


@hmain(base_cfg_dir="cfg", template_cfg_file="cfg/dataset_test.yaml")
def dataset_test() -> None:
    data_module = DataModule()
    data_module.prepare_data()
    data_module.setup()

    img, sem, dep = data_module.get_train_images()

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(img)
    if sem is not None:
        axs[1].imshow(sem)
    if dep is not None:
        axs[2].imshow(dep)
    plt.show()


if __name__ == "__main__":
    dataset_test()

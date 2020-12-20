from typing import Sequence

import numpy as np  # type: ignore
from PIL.Image import Image  # type: ignore


def img2depth(dep_img: Image) -> np.ndarray:
    depth = np.array(dep_img, dtype=np.float32)
    depth = depth[..., 0] + depth[..., 1] * 256 + depth[..., 2] * 256 * 256
    depth = depth / (256 * 256 * 256 - 1)
    depth = depth.clip(0.01, 1)
    depth = 1 / depth
    return depth


def denormalize(img: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    denorm_img = img
    if len(denorm_img.shape) > 3:
        denorm_img = denorm_img.squeeze()
    denorm_img = img * std
    denorm_img = denorm_img + mean
    return np.uint8(denorm_img * 255)

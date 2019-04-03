from typing import Tuple

import numpy as np


def calculate_padding(image: np.ndarray, scaling_factor: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    def scale(i: int, f: int) -> int:
        return (f - i % f) % f

    x, y = image.shape
    px = scale(x, scaling_factor)
    py = scale(y, scaling_factor)

    pad = ((px, 0), (py, 0))
    return pad

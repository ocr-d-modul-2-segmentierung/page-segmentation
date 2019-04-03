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


def fgpa_per_class(pred: np.ndarray, mask: np.ndarray, bin: np.ndarray, n_classes: int) -> np.array:
    """
    Calculate per-class foreground pixel accuracy
    :param pred: prediction
    :param mask: expected result
    :param bin: binarized image (1 is foreground)
    :param n_classes: number of classes used
    :return: array of size n_classes, with fgpa for class i at index i (including 0 for not classified)
    """
    pfg = (pred + 1) * bin - 1
    mfg = (mask + 1) * bin - 1
    fg_count = np.count_nonzero(bin)

    def fgpa_single_class(i: int) -> float:
        errors = pfg.size - np.count_nonzero((pfg == i) == (mfg == i))
        return (fg_count - errors) / fg_count

    return np.fromfunction(np.vectorize(fgpa_single_class), [n_classes+1])

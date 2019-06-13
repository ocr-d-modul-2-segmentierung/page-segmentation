from typing import Tuple, List

import numpy as np


def fgpa(pred: np.ndarray, mask: np.ndarray, bin: np.ndarray) -> np.array:
    """
    Calculate foreground pixel accuracy
    :param pred: prediction
    :param mask: expected result
    :param bin: binarized image (1 is foreground)
    :return overall fgpa for the given prediction
    """
    pfg = pred * bin
    mfg = mask * bin
    fg_count = np.count_nonzero(bin)
    return (fg_count - np.count_nonzero(pfg != mfg)) / fg_count


def fgoverlap_per_class(pred: np.ndarray, mask: np.ndarray, bin: np.ndarray, n_classes: int) \
        -> Tuple[List[float], List[int], List[int], List[int]]:
    """
    Calculate per-class foreground overlap
    :param pred: prediction
    :param mask: expected result
    :param bin: binarized image (1 is foreground)
    :param n_classes: number of classes used
    :return: four arrays of size n_classes, with value for class i at index i (including 0 for not classified):
             overlap, true positives, false positives, false negatices
    """
    pfg = (pred + 1) * bin - 1
    mfg = (mask + 1) * bin - 1

    def overlap_class(i: int) -> Tuple[float, int, int, int]:
        actual, expected = (pfg == i).astype(np.uint8), (mfg == i).astype(np.uint8)

        pixels_of_interest = actual + expected
        n_interest = np.count_nonzero(pixels_of_interest)
        if n_interest == 0:
            # class not relevant, overlap undefined
            return np.nan, 0, 0, 0

        fp = np.count_nonzero(actual > expected)
        fn = np.count_nonzero(expected > actual)

        tp = np.count_nonzero(pixels_of_interest == 2)

        assert n_interest == fp + fn + tp

        return tp / (tp + fp + fn), tp, fp, fn

    overlaps, tps, fps, fns = map(list, zip(*[overlap_class(i) for i in range(n_classes + 1)]))
    return overlaps, tps, fps, fns

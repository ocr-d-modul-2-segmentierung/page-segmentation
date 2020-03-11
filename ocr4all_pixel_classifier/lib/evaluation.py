from typing import Tuple

import numpy as np


def count_matches(mask: np.ndarray, pred: np.ndarray, fg: np.ndarray, label: int) \
        -> Tuple[int, int, int]:
    """
    Count true positives, false positives and false negatives.
    :param mask: the ground truth mask (2D array with labels)
    :param pred: the prediction (2D array with labels)
    :param fg: binary image (2D array, 0 is background, 1 is foreground)
    :param label: which label to count
    :return: true positives, false positives, false negatives
    """
    mask_c = mask == label
    pred_c = pred == label
    mask_fg = mask_c[fg]
    pred_fg = pred_c[fg]
    nmask_fg = ~mask_fg
    npred_fg = ~pred_fg
    tp = np.count_nonzero(np.logical_and(mask_fg, pred_fg))
    fp = np.count_nonzero(np.logical_and(mask_fg, npred_fg))
    fn = np.count_nonzero(np.logical_and(nmask_fg, pred_fg))
    return tp, fp, fn


def total_accuracy(mask: np.ndarray, pred: np.ndarray, fg: np.ndarray) -> Tuple[int, int]:
    """
    Calculate total accuracy across all classes
    :param mask: the ground truth mask (2D array with labels)
    :param pred: the prediction (2D array with labels)
    :param fg: binary image (2D array, 0 is background, 1 is foreground)
    :return: accuracy
    """
    equal = (mask == pred)[fg]
    return np.count_nonzero(equal), equal.size


def f1_measures(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    :return: precision, recall, f1
    """
    if tp == 0:
        return 0.0, 0.0, 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall, f1(precision, recall)


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall)

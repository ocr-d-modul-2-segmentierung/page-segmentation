from typing import Tuple, Callable, Generator, TypeVar, Union

import cv2
import numpy as np
from ocr4all_pixel_classifier.lib.cc import cc_bbox_func


def count_matches(mask: np.ndarray, pred: np.ndarray, label: int) \
        -> Tuple[int, int, int]:
    """
    Count true positives, false positives and false negatives.
    :param mask: the ground truth mask (2D array with labels)
    :param pred: the prediction (2D array with labels)
    :param label: which label to count
    :return: true positives, false positives, false negatives
    """
    mask_label = mask == label
    pred_label = pred == label
    tp = np.count_nonzero(np.logical_and(mask_label, pred_label))
    fp = np.count_nonzero(np.logical_and(mask_label, ~pred_label))
    fn = np.count_nonzero(np.logical_and(~mask_label, pred_label))
    return tp, fp, fn


def total_accuracy(mask: np.ndarray, pred: np.ndarray) -> Tuple[int, int]:
    """
    Calculate total accuracy across all classes
    :param mask: the ground truth mask (2D array with labels)
    :param pred: the prediction (2D array with labels)
    :return: numbers of correct and total elements
    """
    equal = (mask == pred)
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


def cc_equal(threshold: float):
    return lambda pred, mask: np.count_nonzero(pred == mask) / np.size(mask) >= threshold


def cc_matching(label: int, threshold_tp: float, threshold_fp: float, threshold_mask: float = None, assume_filtered: bool = False):
    # return (1,0,0) for TP, (0,1,0) for FP, (0,0,1) for FN
    if not threshold_mask:
        threshold_mask = threshold_tp
    def match(mask, pred):
        size = np.size(mask)
        pred_match_fp = np.count_nonzero(pred == label) / size >= threshold_fp
        pred_match_tp = np.count_nonzero(pred == label) / size >= threshold_tp
        mask_match = np.count_nonzero(mask == label) / size >= threshold_mask
        return np.array(
            [int(pred_match_tp and mask_match), int(pred_match_fp and not mask_match), int(mask_match and not pred_match_tp)])

    return match


class ConnectedComponentEval:
    def __init__(self, mask: np.ndarray, prediction: np.ndarray, binary_image: np.ndarray, connectivity=4):
        if binary_image.ndim > 2:
            raise ValueError("Binary image must be 2-dimensional")

        self.mask = mask
        self.pred = prediction
        self.binary_image = binary_image
        self.filtered_label = None
        self.threshold = None

        self.num_labels, self.labels, self.stats, self.centroids = \
            cv2.connectedComponentsWithStats(binary_image.astype("uint8"), connectivity=connectivity)

    def only_label(self, label: int, threshold: float):
        self.filtered_label = label
        self.threshold = threshold
        return self

    def _filter(self, component: Union[int, np.ndarray], bbox):
        if not self.filtered_label:
            return True

        if type(component) is int:
            component = (bbox(self.labels) == component)

        return self._label_ratio(bbox, self.mask, component) >= self.threshold \
               or self._label_ratio(bbox, self.pred, component) > 0

    def _label_ratio(self, bbox, image, component):
        mask = bbox(image)[component]
        matches = np.count_nonzero(mask == self.filtered_label)
        return matches / np.size(mask)

    def _call_masked(self, component: Union[int, np.ndarray], func, bbox):
        if type(component) is int:
            component = (bbox(self.labels) == component)
        return func(bbox(self.mask)[component], bbox(self.pred)[component])

    T = TypeVar('T')

    def run_per_component(self, func: Callable[[np.ndarray, np.ndarray], T]) -> Generator[T, None, None]:
        for i in range(1, self.num_labels):
            bbox = cc_bbox_func(self.stats, i)
            selection = bbox(self.labels) == i
            if self._filter(selection, bbox):
                yield self._call_masked(selection, func, bbox)

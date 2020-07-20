from typing import Callable

import cv2
import numpy as np

from ocr4all_pixel_classifier.lib.dataset import SingleData


def vote_connected_component_class(pred: np.ndarray, data: SingleData) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(data.binary, connectivity=4)

    for i in range(1, num_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        pred_slice = pred[top:top + h, left:left + w]
        mask = (labels[top:top + h, left:left + w] == i)

        prebin = np.reshape((pred_slice + 1) * mask, pred_slice.size)
        bins = np.bincount(prebin)
        maxclass = np.argmax(bins[1:])
        pred[top:top + h, left:left + w] = pred_slice - mask * pred_slice + mask * maxclass

    return pred


def add_bounding_boxes(pred: np.ndarray, data: SingleData) -> np.ndarray:
    classes = np.unique(pred)
    newpred = np.zeros_like(pred)
    for c in classes:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred == c, connectivity=4)

        for i in range(1, num_labels):
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            newpred[top:top + h, left:left + w] = c
    return newpred


def find_postprocessor(key: str) -> Callable[[np.ndarray, SingleData], np.ndarray]:
    return POSTPROCESSORS[key.lower().replace('_', '').replace('-', '')]


def postprocess_help():
    return (
        "Postprocessors available:\n"
        "cc_majority:    classify all pixels of each connected component as most frequent class.\n"
        "bounding_boxes: replace each connected component in the prediction with its bounding box.\n"
    )


POSTPROCESSORS = {
    'ccmajority': vote_connected_component_class,
    'ccvote': vote_connected_component_class,
    'voteconnectedcomponents': vote_connected_component_class,
    'votecomponents': vote_connected_component_class,
    'boundingboxes': add_bounding_boxes,
    'bbox': add_bounding_boxes,
}

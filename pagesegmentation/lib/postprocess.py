import cv2
import numpy as np

from pagesegmentation.lib.dataset import SingleData

def vote_connected_component_class(pred: np.ndarray, data: SingleData) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(data.binary, connectivity=4)

    for i in range(num_labels):
        l = stats[i, cv2.CC_STAT_LEFT]
        t = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        pred_slice = pred[t:t + h, l:l + w]
        mask = (labels[t:t + h, l:l + w] == i)

        prebin = np.reshape((pred_slice + 1) * mask, pred_slice.size)
        bins = np.bincount(prebin)
        maxclass = np.argmax(bins[1:])
        pred[t:t + h, l:l + w] = pred_slice - mask * pred_slice + mask * maxclass

    return pred

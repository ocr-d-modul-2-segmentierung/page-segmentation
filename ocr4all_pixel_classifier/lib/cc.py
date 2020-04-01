import cv2
import numpy as np

def cc_bbox(image: np.ndarray, cc_stats, cc_index):
    left = cc_stats[cc_index, cv2.CC_STAT_LEFT]
    top = cc_stats[cc_index, cv2.CC_STAT_TOP]
    w = cc_stats[cc_index, cv2.CC_STAT_WIDTH]
    h = cc_stats[cc_index, cv2.CC_STAT_HEIGHT]

    return image[top:top + h, left:left + w]

def cc_bbox_func(cc_stats, cc_index):
    left = cc_stats[cc_index, cv2.CC_STAT_LEFT]
    top = cc_stats[cc_index, cv2.CC_STAT_TOP]
    w = cc_stats[cc_index, cv2.CC_STAT_WIDTH]
    h = cc_stats[cc_index, cv2.CC_STAT_HEIGHT]

    return lambda image: image[top:top + h, left:left + w]

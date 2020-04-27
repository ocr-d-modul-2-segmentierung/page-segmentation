#!/usr/bin/env python3

from typing import Dict, Tuple, List

import cv2
import numpy as np

from ocr4all_pixel_classifier.lib.xycut import do_xy_cut, Segment

ColorMapping = Dict[str, np.ndarray]


def seg(left_upper: Tuple[int, int], right_lower: Tuple[int, int]) -> Segment:
    return Segment(left_upper[0], left_upper[1], right_lower[0], right_lower[1])


DEFAULT_COLOR_MAPPING = {
    "image": np.array([0, 255, 0]),
    "text": np.array([0, 0, 255]),
}


def find_segments(orig_height: int, image: np.ndarray, char_height: int, resize_height: int,
                  color_mapping: ColorMapping) -> Tuple[List[Segment], List[Segment]]:
    # Scale image to specific height for more generic usage of threshold values
    scale_percent = resize_height / image.shape[0]
    height = resize_height
    width = int(image.shape[1] * scale_percent)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    image = dilate(image)

    # Get resizing factor of the scaled image compared to the original one (binary)
    absolute_resize_factor = height / orig_height

    # Determines how many pixels in one line/column need to exist to indicate a match
    px_threshold_line = int(char_height * absolute_resize_factor)
    px_threshold_column = int(char_height * absolute_resize_factor)
    # Determines the size of a gap in pixels to split the found matches into segments
    split_size_horizontal = int(char_height * 2 * absolute_resize_factor)
    split_size_vertical = int(char_height * absolute_resize_factor)

    def scale_all(segments, factor):
        return [seg.scale(factor) for seg in segments]

    # Calculate x-y-cut and get its segments
    segments_text = do_xy_cut(image, px_threshold_line, px_threshold_column, split_size_horizontal, split_size_vertical,
                              color_mapping["text"])
    segments_image = do_xy_cut(image, px_threshold_line, px_threshold_column, split_size_horizontal,
                               split_size_vertical,
                               color_mapping["image"])


    return scale_all(segments_text, 1.0 / absolute_resize_factor),\
           scale_all(segments_image, 1.0 / absolute_resize_factor)


def dilate(bin_image: np.ndarray):
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(bin_image, kernel, iterations=1)
    return im

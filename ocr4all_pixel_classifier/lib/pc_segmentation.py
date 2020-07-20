#!/usr/bin/env python3

from typing import Dict, Tuple, List

import cv2
import numpy as np

from ocr4all_pixel_classifier.lib.xycut import do_xy_cut, RectSegment, CVContour

ColorMapping = Dict[str, np.ndarray]


def seg(left_upper: Tuple[int, int], right_lower: Tuple[int, int]) -> RectSegment:
    return RectSegment(left_upper[0], left_upper[1], right_lower[0], right_lower[1])


DEFAULT_COLOR_MAPPING = {
    "image": np.array([0, 255, 0]),
    "text": np.array([0, 0, 255]),
}


def find_segments(orig_height: int, image: np.ndarray, char_height: int, resize_height: int,
                  color_mapping: ColorMapping, only_images=False) \
        -> Tuple[List[RectSegment], List[RectSegment]]:
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
    segments_image = do_xy_cut(image, px_threshold_line, px_threshold_column, split_size_horizontal,
                               split_size_vertical,
                               color_mapping["image"])

    segments_image = scale_all(segments_image, 1.0 / absolute_resize_factor)

    if only_images:
        segments_text = []
    else:
        segments_text = do_xy_cut(image, px_threshold_line, px_threshold_column, split_size_horizontal,
                                  split_size_vertical,
                                  color_mapping["text"])
        segments_text = scale_all(segments_image, 1.0 / absolute_resize_factor)

    return segments_text, segments_image


def dilate(bin_image: np.ndarray):
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(bin_image, kernel, iterations=1)

    return im


def get_text_contours(image, char_height, color_match):
    if type(color_match) is dict:
        color = np.array(color_match["text"])
    else:
        color = np.array(color_match)

    # Extract pixels with the given color from inverted image
    # Foreground pixels are white
    image = cv2.inRange(image, color, color)

    # Noise removal
    # First ensure that all structures are filled to prevent "holes" (closing)
    # Next remove structues below a threshold size as only 1/3 of a character (opening)
    # The kernel parameters need to provide a reliable balance for:
    # text preservation, region separation, and noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(char_height), int(char_height)))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(char_height / 3), int(char_height / 3)))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Use a kernel based on the char_height to analyze the image on character level
    # First build morphological gradient to detected edges of the characters in text regions
    # Next fill the outlined regions with morphological closing

    # First dilate the foreground pixels to create regions for each possible character
    # Next connect the found character regions with
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(char_height / 1.1), int(char_height / 1.1)))
    region_chars = cv2.dilate(image, kernel, iterations=1)
    region_text = cv2.morphologyEx(region_chars, cv2.MORPH_CLOSE, kernel)

    # Change foreground pixles to black to allow contour detection
    image = (255 - image)

    # Find polygons of contiguous text regions in the image
    contours, hierarchy = cv2.findContours(region_text, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    draw_color = color.tolist().reverse()
    for contour in contours:
        # Draw found polygons over binary image
        # fill: To prevent remaining "holes" that are enclosed by a region
        cv2.drawContours(image, [contour], 0, draw_color, cv2.FILLED)

    # Add 1px white border around the image
    # Contours that touch borders of the image cannot be found (OpenCV bug?)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # Extract final segments based on previous created regions
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # First contour equals the whole page, so skip it
    # Reverse list to preserve region ordering
    return list(map(CVContour, contours[1:][::-1]))

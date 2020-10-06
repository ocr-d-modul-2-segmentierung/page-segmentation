import os
from typing import Tuple, Dict, Callable, List

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageType
from ocr4all.files import split_filename
from ocr4all.colors import ColorMap

from ocr4all_pixel_classifier.lib.xycut import RGBColor, AnyRegion, RectSegment, CVContour


# TODO: make this not fixed to two segment classes and images always being rectangles

def render_regions(output_dir: str,
                   extension: str,
                   orig_shape: Tuple[int, int],
                   prediction_path: str,
                   label_colors: ColorMap,
                   method: Callable[
                       [Tuple[int, int], ColorMap, List[AnyRegion], List[AnyRegion]], ImageType],
                   segments_text: List[AnyRegion],
                   segments_image: List[AnyRegion],
                   ):
    mask_image = method(orig_shape, label_colors, segments_text, segments_image)
    _, image_basename, _ = split_filename(prediction_path)
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, image_basename + "." + extension)
    print(f"Saving to {outfile}")
    mask_image.save(outfile)


def render_xycut(orig_shape: Tuple[int, int], label_colors: ColorMap,
                 segments_text: List[RectSegment], segments_image: List[RectSegment]):
    size=tuple(reversed(orig_shape))
    mask_image = render_rect_segments(size, [
        (label_colors.color_for_label('text'), segments_text),
        (label_colors.color_for_label('image'), segments_image),
    ])
    return mask_image


def render_morphological(orig_shape: Tuple[int, int], label_colors: ColorMap,
                         segments_text: List[CVContour], segments_image: List[RectSegment]):
    mask_image = render_rect_segments(orig_shape, [(label_colors.color_for_label('image'), segments_image)])
    mask_image = render_ocv_contours(mask_image, segments_text, label_colors.color_for_label('text'))
    return mask_image


def render_rect_segments(size: Tuple[int, int], segment_groups: List[Tuple[RGBColor, List[RectSegment]]],
                         base_color: Tuple[int, int, int] = (255, 255, 255)) -> ImageType:
    pil_image = Image.new('RGB', size, base_color)
    canvas = ImageDraw.Draw(pil_image)
    for color, segments in segment_groups:
        for seg in segments:
            canvas.rectangle(seg.as_xy(), fill=color, outline=color)
    return pil_image


def render_ocv_contours(base_image: ImageType, contours: List[CVContour], color_rgb: RGBColor):
    color_bgr = np.array(color_rgb).tolist()
    image_arr = np.asarray(base_image)
    cv2.drawContours(image_arr, list(map(lambda c: c.contour, contours)), -1, color_bgr, cv2.FILLED)
    return Image.fromarray(image_arr)

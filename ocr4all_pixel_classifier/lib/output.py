from dataclasses import dataclass
from typing import Optional

import numpy as np
import os

from ocr4all_pixel_classifier.lib.dataset import SingleData


@dataclass
class Masks:
    color: np.ndarray
    overlay: np.ndarray
    inverted_overlay: np.ndarray
    fg_color_mask: Optional[np.ndarray] = None


def output_data(output_dir, pred, data: SingleData, color_map):
    if len(pred.shape) == 3:
        assert (pred.shape[0] == 1)
        pred = pred[0]

    if data.output_path:
        filename = data.output_path
        dir = os.path.dirname(filename)
        if os.path.isabs(dir):
            os.makedirs(dir, exist_ok=True)
        elif dir:
            for category in ["color", "overlay", "inverted"]:
                os.makedirs(os.path.join(output_dir, category, dir), exist_ok=True)
    else:
        filename = os.path.basename(data.image_path)

    masks = generate_output_masks(data, pred, color_map)

    from skimage.io import imsave
    imsave(os.path.join(output_dir, "color", filename), masks.color)
    imsave(os.path.join(output_dir, "overlay", filename), masks.overlay)
    imsave(os.path.join(output_dir, "inverted", filename), masks.inverted_overlay)


def generate_output_masks(data, pred, color_map) -> Masks:
    from ocr4all_pixel_classifier.lib.dataset import label_to_colors
    color_mask = label_to_colors(pred, colormap=color_map)
    foreground = np.stack([(1 - data.binary)] * 3, axis=-1)
    inv_binary = data.binary
    inv_binary = np.stack([inv_binary] * 3, axis=-1)
    overlay_mask = color_mask.copy()
    overlay_mask[foreground == 0] = 0
    inverted_overlay_mask = color_mask.copy()
    inverted_overlay_mask[inv_binary == 0] = 0
    fg_color_mask = color_mask.copy()
    fg_color_mask[foreground != 0] = 0

    return Masks(
        color=color_mask,
        overlay=overlay_mask,
        inverted_overlay=inverted_overlay_mask,
        fg_color_mask=fg_color_mask,
    )


def scale_to_original_shape(data, pred):
    from ocr4all_pixel_classifier.lib.util import preserving_resize
    resized_image = preserving_resize(data.image, data.original_shape)
    pred = preserving_resize(pred, data.original_shape).astype('int64')
    from dataclasses import replace
    data = replace(data,
                   binary=data.orig_binary,
                   image=resized_image
                   )
    return data, pred

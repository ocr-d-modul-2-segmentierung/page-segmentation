import glob
import os
from random import shuffle
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image


def gray_to_rgb(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        img = img[..., np.newaxis]
        return np.concatenate(3 * (img,), axis=-1)
    else:
        return img


def image_to_batch(img):
    if len(img.shape) == 2:
        return np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
    else:

        assert img.shape != 3
        return np.expand_dims(img, axis=0)


def imread(path, as_gray=False):
    """
    Read RGB image, remove an eventual alpha channel, and convert to numpy
    """
    pil_image = Image.open(path)
    if as_gray:
        pil_image = pil_image.convert('L')
    else:
        pil_image = pil_image.convert('RGB')
    return np.asarray(pil_image)


def imread_bin(path, white_is_fg=False):
    """
    Read a binary image, where the returned Matrix contains True for foreground
    pixels and false for background pixels.
    By default, black is assumed as foreground.
    :param: white_is_fg invert image
    """

    pil_image = Image.open(path)
    bin = np.asarray(pil_image.convert('1'))
    if white_is_fg:
        return bin
    else:
        return np.logical_not(bin)


def match_filenames(base_files: List[str], *file_lists: str) -> Tuple[bool, Optional[str]]:
    """
    Compares filenames in given lists to check if they match up. This requires all filenames in file_lists to start with
    the basename of the corresponding element in base_file. If the file names do not match, or the lists have different
    lengths, False is returned along with an error message.
    :param base_files: file names which are used for prefix check (i.e. use files without _MASK, .bin, etc. here)
    :param file_lists: other file names with may have additional suffixes
    :return: result of check along with error message if failed.
    """
    for list in file_lists:
        if len(list) != len(base_files):
            return False, "List length doesn't match"

    for entry in zip(base_files, *file_lists):
        first = os.path.basename(entry[0])
        first_root, _ = os.path.splitext(first)
        for n in entry[1:]:
            next = os.path.basename(n)
            if not next.startswith(first_root):
                return False, "filename mismatch ({} ≠ {})".format(first, next)
    return True, None


def preserving_resize(image: np.ndarray, target_shape) -> np.ndarray:
    """
    Resizes given image to target_shape while preserving values (i.e. no anti aliasing or range change)
    :param image: input image
    :param target_shape: target_shape
    :return: resized array
    """
    from skimage.transform import resize
    return resize(image, target_shape, order=0, anti_aliasing=False, preserve_range=True)


def glob_all(filenames):
    return [g for f in filenames for g in glob.glob(f)]


def split_filename(image) -> Tuple[str, str, str]:
    image = os.path.basename(image)
    dir = os.path.dirname(image)
    base, ext = image.split(".", 1)
    return dir, base, ext


def random_indices(lst) -> List[int]:
    indices = list(range(len(lst)))
    shuffle(indices)
    return indices


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
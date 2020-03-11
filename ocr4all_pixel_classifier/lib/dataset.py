import json
import multiprocessing
import os
from dataclasses import dataclass
from random import shuffle
from typing import List, Tuple, Optional, Any

import numpy as np
import tqdm
from PIL import Image
from skimage.transform import resize, rescale

from ocr4all_pixel_classifier.lib.image_map import rgb_to_label
from ocr4all_pixel_classifier.lib.util import imread


@dataclass
class SingleData:
    image: np.ndarray = None
    binary: Optional[np.ndarray] = None
    mask: np.ndarray = None
    image_path: Optional[str] = None
    binary_path: Optional[str] = None
    mask_path: Optional[str] = None
    line_height_px: Optional[int] = 1
    original_shape: Tuple[int, int] = None
    output_path: Optional[str] = None
    user_data: Any = None


@dataclass
class Dataset:
    data: List[SingleData]
    color_map: dict

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self.data.__iter__()


def list_dataset(root_dir, line_height_px=None, binary_dir_="binary_images", images_dir_="images", masks_dir_="masks",
                 masks_postfix="", normalizations_dir="normalizations",
                 verify_filenames=False):
    def listdir(dir, postfix="", not_postfix=False):
        if len(postfix) > 0 and not_postfix:
            return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if not f.endswith(postfix)]
        else:
            return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if f.endswith(postfix)]

    def extract_char_height(file):
        with open(file, 'r') as f:
            return json.load(f)["char_height"]

    def if_startswith(fn, sw):

        return [b for b in fn if any([os.path.basename(b).startswith(s) for s in sw])]

    binary_dir = os.path.join(root_dir, binary_dir_)
    images = os.path.join(root_dir, images_dir_)
    masks = os.path.join(root_dir, masks_dir_)

    for d in [root_dir, binary_dir, images, masks]:
        if not os.path.exists(d):
            raise Exception("Dataset dir does not exist at '%s'" % d)

    bin, img, m = listdir(binary_dir), listdir(images, masks_postfix, True), listdir(masks, masks_postfix)
    if verify_filenames:
        def filenames(fn, postfix=None):
            if postfix and len(postfix) > 0:
                fn = [f[:-len(postfix)] if f.endswith(postfix) else f for f in fn]
                return {os.path.basename(f).split('.')[0]: f + postfix for f in fn}

            x = {os.path.basename(f).split('.')[0]: f for f in fn}
            return x

        bin_dir = filenames(bin)
        img_dir = filenames(img)
        mask_dir = filenames(m, masks_postfix)
        base_names = set(bin_dir.keys()).intersection(set(img_dir.keys())).intersection(
            set(mask_dir.keys()))

        bin = [bin_dir.get(basename) for basename in base_names]
        img = [img_dir.get(basename) for basename in base_names]
        m = [mask_dir.get(basename) for basename in base_names]

    else:
        base_names = None

    if not line_height_px:
        norm_dir = os.path.join(root_dir, normalizations_dir)
        if not os.path.exists(norm_dir):
            raise Exception("Norm dir does not exist at '{}'".format(norm_dir))

        line_height_px = listdir(norm_dir)
        if verify_filenames:
            line_height_px = if_startswith(line_height_px, base_names)
        line_height_px = list(map(extract_char_height, line_height_px))
        assert (len(line_height_px) == len(m))
    else:
        line_height_px = [line_height_px] * len(m)

    if len(bin) == len(img) and len(img) == len(m):
        pass
    else:
        raise Exception("Mismatch in dataset files length: %d, %d, %d!" % (len(bin), len(img), len(m)))

    return [{"binary_path": b_p, "image_path": i_p, "mask_path": m_p, "line_height_px": l_h}
            for b_p, i_p, m_p, l_h in zip(bin, img, m, line_height_px)]


#DEPRECATED
def color_to_label(mask, colormap: dict):
    out = np.zeros(mask.shape[0:2], dtype=np.int32)

    if mask.ndim == 2:
        return mask.astype(np.int32) / 255
    if mask.shape[2] == 2:
        return mask[:, :, 0].astype(np.int32) / 255
    mask = mask.astype(np.uint32)
    mask = 256 * 256 * mask[:, :, 0] + 256 * mask[:, :, 1] + mask[:, :, 2]
    for color, label in colormap.items():
        color_1d = 256 * 256 * color[0] + 256 * color[1] + color[2]
        out += (mask == color_1d) * label[0]
    return out


def label_to_colors(mask, colormap: dict):
    out = np.zeros(mask.shape + (3,), dtype=np.int64)
    for color, label in colormap.items():
        trues = np.stack([(mask == label[0])] * 3, axis=-1)
        out += np.tile(color, mask.shape + (1,)) * trues

    out = np.ndarray.astype(out, dtype=np.uint8)
    return out


def scale_binary(binary: np.ndarray, scale: float):
    return rescale(binary, scale,
                   order=0,
                   anti_aliasing=False,
                   preserve_range=True,
                   multichannel=False)


def scale_image(img, target_shape):
    return resize(
        img,
        target_shape,
        order=3,
        anti_aliasing=len(np.unique(img)) > 2,
        preserve_range=True)


def prepare_images(image: np.ndarray, binary: np.ndarray, target_line_height: int, line_height_px: int,
                   max_width: Optional[int] = None):
    scale = target_line_height / line_height_px

    bin = binary / 255 if np.max(binary) > 1 else binary
    bin = 1.0 - scale_binary(bin, scale)
    img = 1.0 - scale_image(image, bin.shape) / 255

    if max_width is not None:
        n_scale = max_width / bin.shape[1]
        if n_scale < 1.0:
            bin = scale_binary(bin, n_scale)
            img = scale_image(img, bin.shape)

    img = (img * 255).astype(np.uint8)
    bin = bin.astype(np.uint8)
    return img, bin


class DatasetLoader:
    def __init__(self, target_line_height, color_map, prediction=False, max_width=None):
        self.target_line_height = target_line_height
        self.prediction = prediction
        self.color_map = color_map
        self.max_width = max_width

    def load_images(self, dataset_file_entry: SingleData) -> SingleData:
        def load_if_needed(data: SingleData, attr: str, as_gray: bool) -> np.ndarray:
            file = getattr(data, attr)
            if file is not None:
                return file
            else:
                return imread(getattr(data, attr + '_path'), as_gray=as_gray)

        # inverted grayscale (black background)
        img = load_if_needed(dataset_file_entry, 'image', as_gray=True)

        original_shape = img.shape
        bin = load_if_needed(dataset_file_entry, 'binary', as_gray=True)

        img, bin = prepare_images(img, bin, self.target_line_height, dataset_file_entry.line_height_px, self.max_width)

        scaled_shape = img.shape

        # color
        if not self.prediction:
            mask = load_if_needed(dataset_file_entry, 'mask', as_gray=False)
            mask = resize(mask, scaled_shape, order=0, anti_aliasing=False, preserve_range=True)

            if mask.ndim == 3:
                mask = rgb_to_label(mask, self.color_map)
            elif mask.ndim == 2:
                u_values = np.unique(mask)
                for ind, x in enumerate(u_values):
                    mask[mask == x] = ind
            assert (mask.shape == img.shape)
            dataset_file_entry.mask = mask.astype(np.uint8)

        dataset_file_entry.binary = bin
        dataset_file_entry.image = img
        dataset_file_entry.original_shape = original_shape

        return dataset_file_entry

    def load_data(self, all_dataset_files) -> Dataset:

        with multiprocessing.Pool(processes=12, maxtasksperchild=100) as p:
            out = list(tqdm.tqdm(p.imap(self.load_images, all_dataset_files), total=len(all_dataset_files)))

        return Dataset(out, self.color_map)

    def load_data_from_json(self, files, type) -> Dataset:
        all_files = []
        for f in files:
            all_files += [SingleData(**d) for d in json.load(open(f, 'r'))[type]]
        print("Loading {} data of type {}".format(len(all_files), type))
        return self.load_data(all_files)

    # noinspection PyUnusedLocal,PyPep8Naming
    def load_test(self):
        dataset_root = "/scratch/Datensets_Bildverarbeitung/page_segmentation"

        ds_ocr_d = ("OCR-D", None)
        ds1 = ("GW5060", 31)
        ds2 = ("GW5064", 36)
        ds3 = ("GW5066", 18)

        all_train_files, all_test_files = [], []
        for dataset, scale in [ds_ocr_d]:
            all_train_files += list_dataset(os.path.join(dataset_root, dataset), scale)

        GW5064_data_files = list_dataset(os.path.join(dataset_root, "GW5064"), line_height_px=36)
        OCR_D_data_files = list_dataset(os.path.join(dataset_root, "OCR-D"), line_height_px=None)
        indices = list(range(len(GW5064_data_files)))
        shuffle(indices)
        train_indices = indices[:20]
        test_indices = indices[20:]

        # all_train_data = load_data([GW5064_data_files[i] for i in train_indices])
        # all_test_data = load_data([GW5064_data_files[i] for i in test_indices])

        all_train_data = self.load_data(OCR_D_data_files)
        all_test_data = []

        # all_train_data = load_data(all_train_files[:])
        # all_test_data = load_data(all_test_files[:])

        return all_train_data, all_test_data


if __name__ == "__main__":
    loader = DatasetLoader(4, color_map={})
    loader.load_test()

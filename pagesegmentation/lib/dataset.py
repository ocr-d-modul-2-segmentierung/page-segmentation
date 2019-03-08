import multiprocessing

import json
import numpy as np
import os
import scipy.misc as misc
import scipy.ndimage as ndimage
import tqdm
from dataclasses import dataclass
from random import shuffle
from typing import List, Tuple, Optional, Any

from pagesegmentation.lib.image_ops import calculate_padding


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
    xpad: Optional[int] = 0
    ypad: Optional[int] = 0
    user_data: Any = None


@dataclass
class Dataset:
    data: List[SingleData]

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
            return [os.path.splitext(os.path.basename(f))[0] for f in fn]

        base_names = set(filenames(bin)).intersection(set(filenames(img))).intersection(
            set(filenames(m, masks_postfix)))

        bin = if_startswith(bin, base_names)
        img = if_startswith(img, base_names)
        m = if_startswith(m, base_names)

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



def color_to_label(mask, marginalia_as_text=True):
    color_label_pairs = [
        ((0, 0, 0),           0),
        ((255, 255, 255),     0),
        ((255, 0, 0),         1),
        ((0, 255, 0),         2),
        ((0, 0, 255),         3),
        ((255, 0, 255),       1),
        ((255, 255, 0),       1 if marginalia_as_text else 3),
        ((128, 0, 0),         1),
        ((0, 255, 255),       1),
        ((128, 128, 0),       3),  # headline
        ((0, 128, 128),       2),  # drop capital
    ]

    out = np.zeros(mask.shape[0:2], dtype=np.int32)
    mask = mask.astype(np.uint32)
    mask = 256 * 256 * mask[:, :, 0] + 256 * mask[:, :, 1] + mask[:, :, 2]

    for color, label in color_label_pairs:
        color = 256 * 256 * color[0] + 256 * color[1] + color[2]
        out += (mask == color) * label

    return out


def label_to_colors(mask):
    colors = [
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
    ]
    labels = [
        0,
        1,
        2,
        3,
    ]
    out = np.zeros(mask.shape + (3,), dtype=np.int64)
    for color, label in zip(colors, labels):
        trues = np.stack([(mask == label)] * 3, axis=-1)
        out += np.tile(color, mask.shape + (1,)) * trues

    out = np.ndarray.astype(out, dtype=np.uint8)
    return out


class DatasetLoader:
    def __init__(self, target_line_height, prediction=False):
        self.target_line_height = target_line_height
        self.prediction = prediction

    def load_images(self, dataset_file_entry: SingleData) -> SingleData:
        scale = self.target_line_height / dataset_file_entry.line_height_px

        # inverted grayscale (black background)
        img = dataset_file_entry.image if dataset_file_entry.image is not None else ndimage.imread(dataset_file_entry.image_path, flatten=True)
        original_shape = img.shape
        bin = dataset_file_entry.binary if dataset_file_entry.binary is not None else ndimage.imread(dataset_file_entry.binary, flatten=True)
        bin = 1.0 - misc.imresize(bin, scale, interp="nearest") / 255
        img = 1.0 - misc.imresize(img, bin.shape, interp="lanczos") / 255

        f = 2 ** 3
        pad = calculate_padding(bin, f)
        img = np.pad(img, pad, 'edge')
        bin = np.pad(bin, pad, 'edge')

        def check(i):
            if i.shape[0] % f != 0 or i.shape[1] % f != 0:
                raise Exception(
                    "Padding not working. Output shape ({}x{}) should be divisible by {}. Dataset entry: {}".format(
                        i.shape[0], i.shape[1], f, dataset_file_entry))

        check(img)
        check(bin)

        # color
        if not self.prediction:
            mask = dataset_file_entry.mask if dataset_file_entry.mask is not None else ndimage.imread(dataset_file_entry.mask_path, flatten=False)
            mask = misc.imresize(mask, bin.shape, interp='nearest')
            if mask.ndim == 3:
                mask = color_to_label(mask)
            mean = np.mean(mask)
            if not 0 <= mean < 3:
                raise Exception("Invalid file at {}".format(dataset_file_entry))

            mask = np.pad(mask, pad, 'edge')

            check(mask)
            assert (mask.shape == img.shape)
            dataset_file_entry.mask = mask.astype(np.uint8)

        dataset_file_entry.binary = bin.astype(np.uint8)
        dataset_file_entry.image = (img * 255).astype(np.uint8)
        dataset_file_entry.original_shape = original_shape
        dataset_file_entry.xpad = pad[0][0]
        dataset_file_entry.ypad = pad[1][0]

        return dataset_file_entry

    def load_data(self, all_dataset_files) -> Dataset:
        with multiprocessing.Pool(processes=12, maxtasksperchild=100) as p:
            out = list(tqdm.tqdm(p.imap(self.load_images, all_dataset_files), total=len(all_dataset_files)))

        return Dataset(out)

    def load_data_from_json(self, files, type) -> Dataset:
        all_files = []
        for f in files:
            all_files += [SingleData(**d) for d in json.load(open(f, 'r'))[type]]

        print("Loading {} data of type {}".format(len(all_files), type))
        return self.load_data(all_files)

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
    loader = DatasetLoader(4)
    loader.load_test()

    working_dir = "/scratch/wick/datasets/page_segmentation/Prediction/FCN_tf/train_ocr-d_test_5066"

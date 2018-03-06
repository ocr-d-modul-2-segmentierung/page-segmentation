import os
import numpy as np
import scipy.ndimage as ndimage
import scipy.misc as misc
import multiprocessing
import tqdm
import skimage.transform as img_trans
import json
from random import shuffle


def list_dataset(root_dir, line_height_px=None):
    def listdir(d):
        return [os.path.join(d, f) for f in sorted(os.listdir(d))]

    def extract_char_height(file):
        with open(file, 'r') as f:
            return json.load(f)["char_height"]

    binary_dir = os.path.join(root_dir, "binary_images")
    images = os.path.join(root_dir, "images")
    masks = os.path.join(root_dir, "masks")

    for d in [root_dir, binary_dir, images, masks]:
        if not os.path.exists(d):
            raise Exception("Dataset dir does not exist at '%s'" % d)

    bin, img, m = listdir(binary_dir), listdir(images), listdir(masks)

    if not line_height_px:
        norm_dir = os.path.join(root_dir, "normalizations")
        if not os.path.exists(norm_dir):
            raise Exception("Norm dir does not exist at '{}'".format(norm_dir))

        line_height_px = listdir(norm_dir)
        line_height_px = list(map(extract_char_height, line_height_px))
        assert(len(line_height_px) == len(m))
    else:
        line_height_px = [line_height_px] * len(m)


    if len(bin) == len(img) and len(img) == len(m):
        pass
    else:
        raise Exception("Mismatch in dataset files length: %d, %d, %d!" % (len(bin), len(img), len(m)))

    return [{"binary_path": b_p, "image_path": i_p, "mask_path": m_p, "line_height_px": l_h}
            for b_p, i_p, m_p, l_h in zip(bin, img, m, line_height_px)]



def color_to_label(mask):
    colors = [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (255, 255, 0),
        (128, 0, 0),
        (0, 255, 255),
    ]
    labels = [
        0,
        0,
        1,
        2,
        1,
        1,
        1,
        1,
        1,
    ]

    out = np.zeros(mask.shape[0:2], dtype=np.int32)
    mask = 256 * 256 * mask[:, :, 0] + 256 * mask[:, :, 1] + mask[:, :, 2]

    for color, label in zip(colors, labels):
        color = 256 * 256 * color[0] + 256 * color[1] + color[2]
        out += (mask == color) * label

    return out


def label_to_colors(mask):
    colors = [
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
    ]
    labels = [
        0,
        1,
        2,
    ]
    out = np.zeros(mask.shape + (3,), dtype=np.int64)
    for color, label in zip(colors, labels):
        trues = np.stack([(mask == label)] * 3, axis=-1)
        out += np.tile(color, mask.shape + (1,)) * trues

    out = np.ndarray.astype(out, dtype=np.uint8)
    return out


def load_images(dataset_file_entry):
    target_line_height = 7
    scale = target_line_height / dataset_file_entry["line_height_px"]

    # inverted grayscale (black background)
    bin = 1.0 - misc.imresize(ndimage.imread(dataset_file_entry["binary_path"], flatten=True), scale, interp="nearest") / 255
    img = 1.0 - misc.imresize(ndimage.imread(dataset_file_entry["image_path"], flatten=True), scale, interp="lanczos") / 255

    # color
    mask = color_to_label(misc.imresize(ndimage.imread(dataset_file_entry["mask_path"], flatten=False), scale, interp="nearest"))

    x, y = mask.shape

    tx = ((x // 2) // 2) * 4
    ty = ((y // 2) // 2) * 4

    if x % 4 != 0:
        px = tx - x + 4
        x = x + px
    else:
        px = 0

    if y % 4 != 0:
        py = ty - y + 4
        y = y + py
    else:
        py = 0

    pad = ((px, 0), (py, 0))
    mask = np.pad(mask, pad, 'edge')
    img = np.pad(img, pad, 'edge')
    bin = np.pad(bin, pad, 'edge')

    def check(i):
        assert(i.shape[0] % 4 == 0)
        assert(i.shape[1] % 4 == 0)

    check(mask)
    check(img)
    check(bin)

    assert(mask.shape == img.shape)

    dataset_file_entry["binary"] = bin
    dataset_file_entry["image"] = img
    dataset_file_entry["mask"] = mask

    return dataset_file_entry


def load_data(all_dataset_files):
    with multiprocessing.Pool(processes=12) as p:
        out = list(tqdm.tqdm(p.imap(load_images, all_dataset_files), total=len(all_dataset_files)))

    return out

def load_test():
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

    all_train_data = load_data(OCR_D_data_files)
    all_test_data = []

    # all_train_data = load_data(all_train_files[:])
    # all_test_data = load_data(all_test_files[:])

    return all_train_data, all_test_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    load_test()


    working_dir = "/scratch/wick/datasets/page_segmentation/Prediction/FCN_tf/train_ocr-d_test_5066"

import itertools
import json
import multiprocessing
import os
import random
from ast import literal_eval

import numpy as np
import tqdm

from ocr4all_pixel_classifier.lib.util import imread

DEFAULT_IMAGE_MAP = {(255, 255, 255): [0, 'bg'],
                     (255, 0, 0): [1, 'text'],
                     (0, 255, 0): [2, 'image']}

DEFAULT_REVERSE_IMAGE_MAP = {v[1]: np.array(k) for k, v in DEFAULT_IMAGE_MAP.items()}


def get_image_file_colors(path_to_mask: str):
    mask = imread(path_to_mask)
    return get_image_colors(mask)


def get_image_colors(mask: np.ndarray):
    if mask.ndim == 2 or mask.shape[2] == 2:
        return [(255, 255, 255), (0, 0, 0)]

    return np.unique(mask.reshape(-1, mask.shape[2]), axis=0)


def compute_image_map(input_dir, output_dir, max_images=-1, processes=4):
    if not os.path.exists(input_dir):
        raise Exception("Cannot open {}".format(input_dir))
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    if max_images > 0:
        files = random.sample(files, max_images)

    with multiprocessing.Pool(processes=processes) as p:
        colors = [v for v in
                  tqdm.tqdm(p.imap(get_image_file_colors, files), total=len(files))
                  ]
    colors = set(itertools.chain.from_iterable(colors))
    colors = sorted(colors, key=lambda element: (element[0], element[1], element[2]))[::-1]
    color_dict = {str(key): (value, "label") for (value, key) in enumerate(colors)}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'image_map.json'), 'w') as fp:
        json.dump(color_dict, fp)


def load_image_map_from_file(path):
    if not os.path.exists(path):
        raise Exception("Cannot open {}".format(path))

    with open(path) as f:
        data = json.load(f)
    color_map = {literal_eval(k): v for k, v in data.items()}
    return color_map


def rgb_to_label(mask: np.ndarray, image_map: dict):
    out = np.zeros(mask.shape[0:2], dtype=np.int32)

    if mask.ndim == 2 or mask.shape[2] == 2:
        raise ValueError("mask must be an RGB image")

    mask = mask.astype(np.uint32)
    mask = 256 * 256 * mask[:, :, 0] + 256 * mask[:, :, 1] + mask[:, :, 2]
    for color, label in image_map.items():
        color_1d = 256 * 256 * color[0] + 256 * color[1] + color[2]
        out += (mask == color_1d) * label[0]
    return out

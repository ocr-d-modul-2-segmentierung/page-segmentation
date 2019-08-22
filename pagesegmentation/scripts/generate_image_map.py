import multiprocessing
from ast import literal_eval
import argparse
import json
import numpy as np
import os
import tqdm
from PIL import Image
import itertools


def get_image_colors(path_to_mask: np.array):
    image_pil = Image.open(path_to_mask)
    if image_pil.mode == 'RGBA':
        image_pil = image_pil.convert('RGB')
    mask = np.asarray(image_pil)
    if mask.ndim == 2 or mask.shape[2] == 2:
        return [(255, 255, 255), (0, 0, 0)]
    tt = mask.view()
    tt.shape = -1, 3
    height, width, depth = mask.shape
    ifl = tt[..., 0].astype(np.int) * height * width + tt[..., 1].astype(np.int) * width + tt[..., 2].astype(np.int)
    colors = np.unique(ifl, return_inverse=False)
    colors = [(int(color / (height * width)), int((color / width) % height), int(color % width)) for color in colors]
    return colors


def compute_image_map(input_dir, output_dir):
    if not os.path.exists(input_dir):
        raise Exception("Cannot open {}".format(input_dir))

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    # files = files[:10]

    with multiprocessing.Pool(processes=1) as p:
        colors = [v for v in
                        tqdm.tqdm(p.imap(get_image_colors, files), total=len(files))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Mask directory to process")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output dir for the color map")

    args = parser.parse_args()
    compute_image_map(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

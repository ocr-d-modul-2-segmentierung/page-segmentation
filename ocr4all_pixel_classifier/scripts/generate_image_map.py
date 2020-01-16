import multiprocessing
from ast import literal_eval
import argparse
import json
import numpy as np
import os
import tqdm
from PIL import Image
import itertools
import random


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


def compute_image_map(input_dir, output_dir, max_images=-1, processes=4):
    if not os.path.exists(input_dir):
        raise Exception("Cannot open {}".format(input_dir))
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    if max_images > 0:
        files = random.sample(files, max_images)

    with multiprocessing.Pool(processes=processes) as p:
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
    parser = argparse.ArgumentParser(add_help=False)
    paths_args = parser.add_argument_group("Paths")
    paths_args.add_argument("-I", "--input-dir", type=str, required=True,
                            help="Mask directory to process")
    paths_args.add_argument("-O", "--output-dir", type=str, required=True,
                            help="The output dir for the color map")

    opt_args = parser.add_argument_group("optional arguments")
    opt_args.add_argument("-h", "--help", action="help", help="show this help message and exit")
    opt_args.add_argument("--max-image", type=int, default=-1,
                        help="Max images to check for color. -1 to check every mask")
    opt_args.add_argument("-j", "--jobs", "--threads", metavar='THREADS', dest='threads',
                           type=int, default=multiprocessing.cpu_count(),
                           help="Number of threads to use")

    args = parser.parse_args()
    compute_image_map(args.input_dir, args.output_dir, args.max_image, args.processes)


if __name__ == '__main__':
    main()

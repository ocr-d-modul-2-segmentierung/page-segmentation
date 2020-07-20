import argparse
import json
import multiprocessing
import os
import sys
from functools import partial
from math import isnan

import numpy as np
import tqdm

from ocr4all_pixel_classifier.lib.image_ops import compute_char_height


def compute_normalizations(input_dir, output_dir=None, inverse=False, average_all=True):
    if not os.path.exists(input_dir):
        raise Exception("Cannot open {}".format(input_dir))

    if not os.path.isdir(input_dir):
        files = [input_dir]
    else:
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    # files = files[:10]

    with multiprocessing.Pool(processes=12) as p:
        char_heights = [v for v in
                        tqdm.tqdm(p.imap(partial(compute_char_height, inverse=inverse), files), total=len(files))
                        ]

    av_height = np.mean([c for c in char_heights if c])
    if isnan(av_height):
        raise Exception("No chars found in dataset")
    if average_all:
        char_heights = [av_height] * len(char_heights)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for file, height in zip(files, char_heights):
        filename, file_extension = os.path.splitext(os.path.basename(file))
        if height is None:
            height = av_height
        if output_dir:
            output_file = os.path.join(output_dir, filename + ".norm")
            with open(output_file, 'w') as f:
                json.dump(
                    {"file": file, "char_height": int(height)},
                    f,
                    indent=4
                )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    paths_args = parser.add_argument_group("Paths")
    paths_args.add_argument("-I", "--input-dir", type=str, required=True,
                            help="Image directory to process")
    paths_args.add_argument("-O", "--output-dir", type=str, required=True,
                            help="The output dir for the normalization data")

    opt_args = parser.add_argument_group("optional arguments")
    opt_args.add_argument("-h", "--help", action="help", help="show this help message and exit")
    opt_args.add_argument("--average-all", "--average_all", action="store_true",
                          help="Use average height over all images.")
    opt_args.add_argument("--inverse", action="store_true",
                          help="use if white is foreground")
    opt_args.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        compute_normalizations(args.input_dir, args.output_dir, args.inverse, args.average_all)
    else:
        try:
            import warnings
            warnings.filterwarnings('ignore')
            compute_normalizations(args.input_dir, args.output_dir, args.inverse, args.average_all)
        except Exception:
            print("Error:", sys.exc_info()[1])


if __name__ == '__main__':
    main()

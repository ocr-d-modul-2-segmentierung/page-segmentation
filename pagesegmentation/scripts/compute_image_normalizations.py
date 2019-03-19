import multiprocessing

import argparse
import cv2
import json
import numpy as np
import os
import tqdm
from functools import partial


def compute_char_height(file_name: str, inverse: bool):
    if not os.path.exists(file_name):
        raise Exception("File does not exist at {}".format(file_name))

    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if inverse:
        img = cv2.subtract(255, img)

    # labeled, nr_objects = ndimage.label(img > 128)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4)

    possible_letter = [False] + [0.5 < (stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]) < 2
                                 and 10 < stats[i, cv2.CC_STAT_HEIGHT] < 60
                                 and 5 < stats[i, cv2.CC_STAT_WIDTH] < 50
                                 for i in range(1, len(stats))]

    for x in np.nditer(labels, op_flags=['readwrite']):
        x[...] = x * possible_letter[x]

    valid_letter_heights = stats[possible_letter, cv2.CC_STAT_HEIGHT]

    valid_letter_heights.sort()
    try:
        mode = valid_letter_heights[int(len(valid_letter_heights) / 2)]
        return mode
    except IndexError:
        return None


def compute_normalizations(input_dir, output_dir, inverse=True, average_all=True):
    if not os.path.exists(input_dir):
        raise Exception("Cannot open {}".format(input_dir))

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    # files = files[:10]

    with multiprocessing.Pool(processes=12) as p:
        char_heights = [v for v in
                        tqdm.tqdm(p.imap(partial(compute_char_height, inverse=inverse), files), total=len(files))
                        ]

    if average_all:
        av_height = np.mean([c for c in char_heights if c])
        char_heights = [av_height] * len(char_heights)

    if output_dir and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file, height in zip(files, char_heights):
        filename, file_extension = os.path.splitext(os.path.basename(file))
        if output_dir:
            output_file = os.path.join(output_dir, filename + ".norm")
            with open(output_file, 'w') as f:
                json.dump(
                    {"file": file, "char_height": int(height)},
                    f,
                    indent=4
                )

        print(file, height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Image directory to process")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output dir for the info files")
    parser.add_argument("--average_all", action="store_true", default=True,
                        help="Average height over all images")
    parser.add_argument("--inverse", action="store_false", default=True)

    args = parser.parse_args()
    compute_normalizations(args.input_dir, args.output_dir, args.inverse, args.average_all)


if __name__ == '__main__':
    main()

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import multiprocessing
import tqdm
import json
from functools import partial

def computeCharHeight(file_name, inverse):
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
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Image directory to process")
    parser.add_argument("--average_all", action="store_true", default=True,
                        help="bla")
    parser.add_argument("--cut_left", type=float, default=0.05)
    parser.add_argument("--cut_right", type=float, default=0.05)
    parser.add_argument("--inverse", action="store_false", default=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output dir for the info files")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise Exception("Cannot open {}".format(args.input_dir))

    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)]

    # files = files[:10]
    global inverse
    inverse  = args.inverse


    with multiprocessing.Pool(processes=12) as p:
        char_heights = [v for v in tqdm.tqdm(p.imap(partial(computeCharHeight, inverse=args.inverse), files), total=len(files))]

    if args.average_all:
        av_height = np.mean([c for c in char_heights if c])
        char_heights = [av_height] * len(char_heights)

    if args.output_dir and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for file, height in zip(files, char_heights):
        filename, file_extension = os.path.splitext(os.path.basename(file))
        if args.output_dir:
            output_file = os.path.join(args.output_dir, filename + ".norm")
            with open(output_file, 'w') as f:
                json.dump(
                    {"file": file, "char_height": int(height)},
                    f,
                    indent=4
                )

        print(file, height)

if __name__ == '__main__':
    main()

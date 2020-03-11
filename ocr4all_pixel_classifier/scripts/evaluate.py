import argparse
import os

import numpy as np
from tqdm import tqdm

from ocr4all_pixel_classifier.lib.evaluation import count_matches, total_accuracy, f1_measures
from ocr4all_pixel_classifier.lib.image_map import rgb_to_label
from ocr4all_pixel_classifier.lib.util import imread
from ocr4all_pixel_classifier.lib.image_map import load_image_map_from_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", type=str, required=True, nargs="+",
                        help="expected masks")
    parser.add_argument("--preds", type=str, required=True, nargs="+",
                        help="prediction results")
    parser.add_argument("--binary", type=str, required=True, nargs="+",
                        help="binary source images")
    parser.add_argument("--color-map-model", type=str, required=False, help="color map used in model")
    parser.add_argument("--color-map-eval", type=str, required=False, help="color map used in evaluation data set")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also output statistics per image")
    args = parser.parse_args()

    if bool(args.color_map_model) != bool(args.color_map_eval):
        parser.error("color map is only useful if given for both model and evaluation data set")

    if len(args.masks) != len(args.preds) or len(args.preds) != len(args.binary):
        parser.error("Number of masks, predictions and binary images not equal ({} / {} / {})"
                     .format(len(args.masks), len(args.preds), len(args.binary)))

    for m, p, b in zip(args.masks, args.preds, args.binary):
        m = os.path.basename(m)
        b = os.path.basename(b)
        p = os.path.basename(p)
        root, ext = os.path.splitext(p)
        if not m.startswith(root):
            print("filename mismatch ({} ≠ {})".format(m, p))
            exit(1)
        if not b.startswith(root):
            print("filename mismatch ({} ≠ {})".format(b, p))
            exit(1)

    text_tpfpfn = np.zeros([3])
    image_tpfpfn = np.zeros([3])
    correct_total = np.zeros([2])

    if args.color_map_model and args.color_map_eval:
        model_map = load_image_map_from_file(args.color_map_model)
        eval_map = load_image_map_from_file(args.color_map_eval)
    else:
        model_map = eval_map = {(255, 255, 255): [0, 'background'],
                                (0, 255, 0): [1, 'text'],
                                (255, 0, 255): [2, 'image']}

    for mask_p, pred_p, bin_p in tqdm(zip(args.masks, args.preds, args.binary)):
        mask = rgb_to_label(imread(mask_p), eval_map)
        pred = rgb_to_label(imread(pred_p), model_map)

        fg = imread(bin_p) == 0

        text_matches = count_matches(mask, pred, fg, 1)
        text_tpfpfn += text_matches

        image_matches = count_matches(mask, pred, fg, 2)
        image_tpfpfn += image_matches

        correct_total += total_accuracy(mask, pred, fg)

        if args.verbose:
            print("T: {:<10} / {:<10} / {:<10} -> Prec: {:<5f}, Rec: {:<5f}, F1{:<5f} {:>20}"
                  .format(*text_matches, *f1_measures(*text_matches), bin_p))
            print("I: {:<10} / {:<10} / / {:<10} -> Prec: {:<5f}, Rec: {:<5f}, F1{:<5f} {:>20}"
                  .format(*image_matches, *f1_measures(*image_matches), bin_p))

    print("\nText:")
    print(format_total(text_tpfpfn))
    print("\nImage:")
    print(format_total(image_tpfpfn))
    print("\nOverall accuracy:")
    print('================================================================\n')
    print(correct_total[0] / correct_total[1])


def format_total(counts):
    ttp, tfp, tfn = counts
    return '================================================================\n' \
           'total:\n' \
           '{:<10} / {:<10} /  {:<10} -> Prec: {:f}, Rec: {:f}, Acc{:f}' \
        .format(ttp, tfp, tfn, *f1_measures(ttp, tfp, tfn))


if __name__ == "__main__":
    main()

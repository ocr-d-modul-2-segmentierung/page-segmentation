import argparse
import math
from typing import List

from ocr4all_pixel_classifier.lib.dataset import list_dataset
from random import shuffle, seed
import json


def main():
    parser = argparse.ArgumentParser(add_help=False)

    paths_args = parser.add_argument_group("Main paths")
    paths_args.add_argument("-D", "--dataset-path", type=str, required=True,
                            help="base path of dataset")
    paths_args.add_argument("-O", "--output-file", type=str, required=True,
                            help="output location for dataset JSON. for generating multiple splits, add {} where the number should be.")

    subpaths_args = parser.add_argument_group(
        "Dataset paths",
        description="Paths relative to --dataset-path for the various types of file included in the dataset")
    subpaths_args.add_argument("--binary-dir", type=str, default="binary",
                               help="directory name of the binary images")
    subpaths_args.add_argument("--images-dir", type=str, default="images",
                               help="directory name of the images on which to train")
    subpaths_args.add_argument("--masks-dir", type=str, default="masks",
                               help="directory name of the masks")
    subpaths_args.add_argument("--masks-postfix", type=str, default="",
                               help="Postfix to distinguish masks and images, if they are in the same directory "
                                    "(including the file extension)")
    subpaths_args.add_argument("--normalizations-dir", type=str, default="norm")

    split_args = parser.add_argument_group(
        "Set split ratios",
        description="integer values are interpreted as absolute number of images to use. Fractions are interpreted as "
                    "percentage of total number of images. Use -1 for remaining images")
    split_args.add_argument("--n-eval", type=float, default=0, help="For final model evaluation")
    split_args.add_argument("--n-train", type=float, default=-1, help="For training")
    split_args.add_argument("--n-test", type=float, default=20, help="For picking the best model")
    split_args.add_argument("--n-splits", type=int, default=None,
                            help="Create a number of splits by dividing the dataset into the given number of parts. Overrides other ratios")

    opt_args = parser.add_argument_group("optional arguments")
    opt_args.add_argument("-h", "--help", action="help", help="show this help message and exit")
    opt_args.add_argument("--seed", type=int, default=67452)
    opt_args.add_argument("--xheight", "--char_height_of_n", type=int, default=None,
                          help="Height of a small character in the main font")
    opt_args.add_argument("--verify-filenames", action="store_true",
                          help="File names in different folders must match. If not specified, files are matched by"
                               "lexical order only.")

    args = parser.parse_args()

    seed(args.seed)

    data_files = list_dataset(args.dataset_path, args.xheight,
                              binary_dir_=args.binary_dir,
                              images_dir_=args.images_dir,
                              masks_dir_=args.masks_dir,
                              masks_postfix=args.masks_postfix,
                              normalizations_dir=args.normalizations_dir,
                              verify_filenames=args.verify_filenames)

    if args.n_splits:
        multi_split(args, data_files)
    else:
        single_split(args, data_files)


def single_split(args, data_files):
    def fraction_or_absolute(part, collection):
        if 0 < part < 1:
            return int(part * len(collection))
        else:
            return int(part)

    args.n_eval = fraction_or_absolute(args.n_eval, data_files)
    args.n_test = fraction_or_absolute(args.n_test, data_files)
    args.n_train = fraction_or_absolute(args.n_train, data_files)
    if sum([args.n_eval < 0, args.n_train < 0, args.n_test < 0]) > 1:
        raise Exception("Only one dataset may get all remaining files")
    if args.n_eval < 0:
        args.n_eval = len(data_files) - args.n_train - args.n_test
    elif args.n_train < 0:
        args.n_train = len(data_files) - args.n_eval - args.n_test
    elif args.n_test < 0:
        args.n_test = len(data_files) - args.n_eval - args.n_train
    if len(data_files) < args.n_eval + args.n_train + args.n_test:
        raise Exception("The dataset consists of {} files, but eval + train + test = {} + {} + {} = {}".format(
            len(data_files), args.n_eval, args.n_train, args.n_test,
            args.n_eval + args.n_train + args.n_test)
        )
    indices = random_indices(data_files)

    eval = [data_files[d] for d in indices[:args.n_eval]]
    train = [data_files[d] for d in indices[args.n_eval:args.n_eval + args.n_train]]
    test = [data_files[d] for d in indices[args.n_eval + args.n_train:args.n_eval + args.n_train + args.n_test]]

    write_json(args.output_file, args.dataset_path, args.seed, train, test, eval)


def multi_split(args, data_files):
    for i, split in enumerate(create_splits(data_files, args.n_splits)):
        train, test = split
        write_json(args.output_file.format(i), args.dataset_path, args.seed, train, test, [])


def write_json(output_file, dataset_path, seed, train, test, eval):
    content = json.dumps({
        "seed": seed,
        "dataset_path": dataset_path,
        "train": train,
        "test": test,
        "eval": eval,
    }, indent=4)
    with open(output_file, "w") as f:
        f.write(content)

        print("File written to {}".format(output_file))


def random_indices(lst) -> List[int]:
    indices = list(range(len(lst)))
    shuffle(indices)
    return indices


def create_splits(data_files: List[str], num_splits: int):
    input = data_files.copy()
    shuffle(input)
    parts = list(chunks(input, math.ceil(len(input) / num_splits)))

    for i in range(num_splits):
        split = []
        for chunk in range(len(parts)):
            if chunk != i:
                split += parts[chunk]
        yield split, parts[i]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    main()

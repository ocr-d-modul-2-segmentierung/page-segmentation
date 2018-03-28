import argparse
from dataset import list_dataset
from random import shuffle, seed
import json

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=67452)
parser.add_argument("--dataset_path", type=str,
                    default="/scratch/Datensets_Bildverarbeitung/page_segmentation/Badius")
parser.add_argument("--output_file", type=str,
                    default="/scratch/Datensets_Bildverarbeitung/page_segmentation/Badius-0-.json")
parser.add_argument("--char_height_of_n", type=int, default=None)
parser.add_argument("--n_eval", type=int, default=0, help="For final model evaluation")
parser.add_argument("--n_train", type=int, default=-1, help="For training")
parser.add_argument("--n_test", type=int, default=20, help="For picking the best model")
parser.add_argument("--binary_dir", type=str, default="binary_images",
                    help="directory name of the binary images")
parser.add_argument("--images_dir", type=str, default="images",
                    help="directory name of the images on which to train")
parser.add_argument("--masks_dir", type=str, default="masks",
                    help="directory name of the masks")

args = parser.parse_args()

seed(args.seed)

data_files = list_dataset(args.dataset_path, args.char_height_of_n,
                          binary_dir_=args.binary_dir,
                          images_dir_=args.images_dir,
                          masks_dir_=args.masks_dir)

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


indices = list(range(len(data_files)))
shuffle(indices)

eval = [data_files[d] for d in indices[:args.n_eval]]
train = [data_files[d] for d in indices[args.n_eval:args.n_eval + args.n_train]]
test = [data_files[d] for d in indices[args.n_eval + args.n_train:args.n_eval + args.n_train + args.n_test]]


content = json.dumps({
    "seed": args.seed,
    "dataset_path": args.dataset_path,
    "eval": eval,
    "train": train,
    "test": test,
}, indent=4)

with open(args.output_file, "w") as f:
    f.write(content)

    print("File written to {}".format(args.output_file))



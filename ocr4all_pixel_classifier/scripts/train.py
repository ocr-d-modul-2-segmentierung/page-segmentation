import argparse
import json
import sys
from os import path
from typing import List
from ocr4all_pixel_classifier.lib.model import Architecture


def main():
    from ocr4all_pixel_classifier.lib.trainer import TrainSettings, Trainer
    from ocr4all_pixel_classifier.lib.metrics import Monitor

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--l-rate", type=float, default=1e-4,
                        help="set learning rate")
    parser.add_argument("-H", "--target-line-height", type=int, default=6,
                        help="Scale the data images so that the line height matches this value")
    parser.add_argument("-O", "--output", type=str, default="./",
                        help="target directory for model and logs")
    parser.add_argument("--load", type=str, default=None,
                        help="load an existing model and continue training")
    parser.add_argument("-E", "--n-epoch", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--display", type=int, default=100,
                        help="number of iterations between displaying status")
    parser.add_argument("-S", "--early-stopping-max-performance-drops", type=int, default=5,
                        help="number of iterations without improvements after which to stop early")
    parser.add_argument("--data-augmentation", action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("-s", "--split-file", type=str,
                        help="Load splits from a json file")
    parser.add_argument("--train", type=str, nargs="*", default=[], help="Dataset for training")
    parser.add_argument("--test", type=str, nargs="*", default=[], help="Dataset used for early stopping")
    parser.add_argument("--eval", type=str, nargs="*", default=[])
    parser.add_argument("--foreground-masks", action="store_true",
                        help="keep only mask parts that are foreground in binary image")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Generate tensorboard logs")
    parser.add_argument("--reduce-lr-on-plateau", action="store_true",
                        help="Reduce learn rate when on plateau")
    parser.add_argument("--color-map", type=str, default="image_map.json",
                        help="color map to load")
    parser.add_argument('--architecture',
                        default=Architecture.FCN_SKIP,
                        const=Architecture.FCN_SKIP,
                        nargs='?',
                        choices=[x.value for x in list(Architecture)],
                        help='Network architecture to use for training')
    parser.add_argument("--gpu-allow-growth", action="store_true",
                        help="set allow_growth option for Tensorflow GPU. Use if getting CUDNN_INTERNAL_ERROR")
    parser.add_argument("--ignore-types-from-file", action="store_true",
                        help="Ignore the train/test/eval associations given in the dataset files, use only splitfile "
                             "or --train / --test / --eval parameters for association")
    # aliases to support legacy names of options
    parser.add_argument("--l_rate", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--target_line_height", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n_epoch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--early_stopping_max_performance_drops", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--data_augmentation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--split_file", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--foreground_masks", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reduce_lr_on_plateau", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--color_map", type=str, default="image_map.json", help=argparse.SUPPRESS)
    parser.add_argument("--gpu_allow_growth", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    def relpaths(basedir: str, files: List[str]) -> List[str]:
        return [x if x[0] == "/" else path.join(basedir, x) for x in files]

    def is_valid_splitfile(json):
        for category in ["train", "test", "eval"]:
            if type(json[category]) == list and len(json[category]) > 0 and type(json[category][0]) != str:
                return False
        return True

    # json file for splits
    if args.split_file:
        with open(args.split_file) as f:
            d = json.load(f)
            if not is_valid_splitfile(d):
                print("Invalid splitfile. Did you pass a dataset file?")
                sys.exit(1)
            reldir = path.dirname(args.split_file)
            args.train += relpaths(reldir, d["train"])
            args.test += relpaths(reldir, d["test"])
            args.eval += relpaths(reldir, d["eval"])

    from ocr4all_pixel_classifier.lib.dataset import DatasetLoader
    from ocr4all_pixel_classifier.lib.image_map import load_image_map_from_file
    from ocr4all_pixel_classifier.lib.metrics import Loss

    image_map = load_image_map_from_file(args.color_map)
    dataset_loader = DatasetLoader(args.target_line_height, image_map)

    train_data = dataset_loader.load_data_from_json(args.train, "all" if args.ignore_types_from_file else "train")
    test_data = dataset_loader.load_data_from_json(args.test, "all" if args.ignore_types_from_file else "test")
    eval_data = dataset_loader.load_data_from_json(args.eval, "all" if args.ignore_types_from_file else "eval")

    settings = TrainSettings(
        n_epoch=args.n_epoch,
        n_classes=len(dataset_loader.color_map),
        l_rate=args.l_rate,
        train_data=train_data,
        validation_data=test_data,
        evaluation_data=eval_data,
        load=args.load,
        loss=Loss.CATEGORICAL_CROSSENTROPY,
        monitor=Monitor.LOSS,
        display=args.display,
        output_dir=args.output,
        early_stopping_max_performance_drops=args.early_stopping_max_performance_drops,
        threads=6,
        compute_baseline=True,
        foreground_masks=args.foreground_masks,
        data_augmentation=args.data_augmentation,
        tensorboard=args.tensorboard,
        reduce_lr_on_plateau=args.reduce_lr_on_plateau,
        gpu_allow_growth=args.gpu_allow_growth,
        architecture=Architecture(args.architecture)
    )
    trainer = Trainer(settings)
    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()

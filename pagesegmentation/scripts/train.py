import argparse
import json
from os import path
from typing import List

import numpy as np
from pagesegmentation.scripts.generate_image_map import load_image_map_from_file
from pagesegmentation.lib.dataset import DatasetLoader


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', ''):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    from pagesegmentation.lib.trainer import TrainSettings, Trainer
    from pagesegmentation.lib.predictor import Predictor, PredictSettings

    parser = argparse.ArgumentParser()
    parser.add_argument("--l_rate", type=float, default=1e-4)
    parser.add_argument("--target_line_height", type=int, default=6,
                        help="Scale the data images so that the line height matches this value")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--early_stopping_max_l_rate_drops", type=int, default=5)
    parser.add_argument("--prediction_dir", type=str)
    parser.add_argument("--data_augmentation", default=False, action="store_true",
                        help="Load splits from a json file")
    parser.add_argument("--split_file", type=str,
                        help="Load splits from a json file")
    parser.add_argument("--train", type=str, nargs="*", default=[])
    parser.add_argument("--test", type=str, nargs="*", default=[],
                        help="Data used for early stopping"
                        )
    parser.add_argument("--eval", type=str, nargs="*", default=[])
    parser.add_argument("--foreground_masks", default=False, action="store_true",
                        help="keep only mask parts that are foreground in binary image")
    parser.add_argument("--tensorboard", type=str2bool, default=False,
                        help="Generate tensorboard logs")
    parser.add_argument("--reduce_lr_on_plateu", type=str2bool, default=True,
                        help="Reducing LR when on plateau")
    parser.add_argument("--color_map", type=str, required=True,
                        help="color_map to load")
    args = parser.parse_args()

    def relpaths(basedir: str, files: List[str]) -> List[str]:
        return [x if x[0] == "/" else path.join(basedir, x) for x in files]

    # json file for splits
    if args.split_file:
        with open(args.split_file) as f:
            d = json.load(f)
            reldir = path.dirname(args.split_file)
            args.train += relpaths(reldir, d["train"])
            args.test += relpaths(reldir, d["test"])
            args.eval += relpaths(reldir, d["eval"])

    image_map = load_image_map_from_file(args.color_map)
    dataset_loader = DatasetLoader(args.target_line_height, image_map)
    train_data = dataset_loader.load_data_from_json(args.train, "train")
    test_data = dataset_loader.load_data_from_json(args.test, "test")
    print(dataset_loader.color_map)

    eval_data = dataset_loader.load_data_from_json(args.eval, "eval")

    settings = TrainSettings(
        n_iter=args.n_iter,
        n_classes=len(dataset_loader.color_map),
        l_rate=args.l_rate,
        train_data=train_data,
        validation_data=test_data,
        evaluation_data= eval_data,
        load=args.load,
        display=args.display,
        output=args.output,
        early_stopping_max_l_rate_drops=args.early_stopping_max_l_rate_drops,
        threads=8,
        foreground_masks=args.foreground_masks,
        data_augmentation=args.data_augmentation,
        tensorboard=args.tensorboard,
        reduce_lr_on_plateu=args.reduce_lr_on_plateu,
    )
    trainer = Trainer(settings)
    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()

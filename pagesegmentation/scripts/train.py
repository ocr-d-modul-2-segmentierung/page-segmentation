import pagesegmentation.lib.model as model
import numpy as np
from pagesegmentation.lib.dataset import DatasetLoader, label_to_colors
import matplotlib.pyplot as plt
import skimage.io as img_io
import os
import tqdm
import argparse
import json


def main():
    from pagesegmentation.lib.trainer import TrainSettings, Trainer
    from pagesegmentation.lib.predictor import Predictor, PredictSettings

    parser = argparse.ArgumentParser()
    parser.add_argument("--l_rate", type=float, default=1e-3)
    parser.add_argument("--l_rate_drop_factor", type=float, default=0.1)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--target_line_height", type=int, default=6,
                        help="Scale the data images so that the line height matches this value")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--early_stopping_test_interval", type=int, default=100)
    parser.add_argument("--early_stopping_max_keep", type=int, default=10)
    parser.add_argument("--early_stopping_max_l_rate_drops", type=int, default=3)
    parser.add_argument("--early_stopping_on_accuracy", default=False, action="store_true")
    parser.add_argument("--prediction_dir", type=str)
    parser.add_argument("--split_file", type=str,
                        help="Load splits from a json file")
    parser.add_argument("--train", type=str, nargs="*", default=[])
    parser.add_argument("--test", type=str, nargs="*", default=[],
                        help="Data used for early stopping"
    )
    parser.add_argument("--eval", type=str, nargs="*", default=[])
    parser.add_argument("--display", type=int, default=100,
                        help="Display training progress each display iterations.")

    args = parser.parse_args()

    # json file for splits
    if args.split_file:
        with open(args.split_file) as f:
            d = json.load(f)
            args.train += d["train"]
            args.test += d["test"]
            args.eval += d["eval"]

    dataset_loader = DatasetLoader(args.target_line_height)
    train_data = dataset_loader.load_data_from_json(args.train, "train")
    test_data = dataset_loader.load_data_from_json(args.train, "test")
    eval_data = dataset_loader.load_data_from_json(args.eval, "eval")

    settings = TrainSettings(
        n_iter=args.n_iter,
        n_classes=args.n_classes,
        l_rate=args.l_rate,
        train_data=train_data,
        validation_data=test_data,
        load=args.load,
        display=args.display,
        output=args.output,
        early_stopping_test_interval=args.early_stopping_test_interval,
        early_stopping_max_keep=args.early_stopping_max_keep,
        early_stopping_on_accuracy=args.early_stopping_on_accuracy,
        threads=8,
    )
    trainer = Trainer(settings)
    trainer.train()

    predict_settings = PredictSettings(
        mode='test',
        n_classes=settings.n_classes,
    )
    predictor = Predictor(predict_settings, trainer.test_net)

    def compute_total(label, data):
        print("Computing total error of {}".format(label))
        total_a, total_fg = predictor.test(data)
        print("%s: Acc=%.5f FgPA=%.5f" % (label, total_a, total_fg))

    compute_total("Test", test_data)


if __name__ == "__main__":
    main()

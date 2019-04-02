import argparse
import glob
import os

import tqdm

from pagesegmentation.lib.dataset import DatasetLoader
from pagesegmentation.lib.predictor import Predictor, PredictSettings


def glob_all(filenames):
    files = []
    for f in filenames:
        files += glob.glob(f)

    return files


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True,
                        help="Model to load")
    parser.add_argument("--target_line_height", type=int, default=6,
                        help="Scale the data images so that the line height matches this value (must be the same as in training)")
    parser.add_argument("--output", required=True,
                        help="Output dir")
    parser.add_argument("--eval", type=str, nargs="*", default=[], help="JSON file with data to use for evaluation")
    parser.add_argument("--keep_low_res", action="store_true",
                        help="keep low resolution prediction instead of rescaling output to orignal image size")
    args = parser.parse_args()

    mkdir(args.output)

    dataset_loader = DatasetLoader(args.target_line_height, prediction=True)
    data = dataset_loader.load_data_from_json(args.eval, "eval")

    print("Creating net")
    settings = PredictSettings(
        mode='meta',
        network=os.path.abspath(args.load),
        output=args.output,
        high_res_output=not args.keep_low_res,
        n_classes=4
    )
    predictor = Predictor(settings)

    print("Starting prediction")
    for i, pred in tqdm.tqdm(enumerate(predictor.predict(data))):
        pass


if __name__ == "__main__":
    main()

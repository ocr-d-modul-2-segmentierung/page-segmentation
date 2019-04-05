import argparse
import os

import tqdm

from pagesegmentation.lib.dataset import DatasetLoader
from pagesegmentation.lib.postprocess import vote_connected_component_class
from pagesegmentation.lib.predictor import Predictor, PredictSettings


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
    parser.add_argument("--cc_majority", action="store_true",
                        help="classify all pixels of each connected component as most frequent class")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    dataset_loader = DatasetLoader(args.target_line_height, prediction=True)
    data = dataset_loader.load_data_from_json(args.eval, "eval")

    post_processors = []
    if args.cc_majority:
        post_processors += [vote_connected_component_class]

    print("Creating net")
    settings = PredictSettings(
        mode='meta',
        network=os.path.abspath(args.load),
        output=args.output,
        high_res_output=not args.keep_low_res,
        n_classes=4,
        post_process=post_processors
    )
    predictor = Predictor(settings)

    print("Starting prediction")
    for _, _ in tqdm.tqdm(enumerate(predictor.predict(data))):
        pass


if __name__ == "__main__":
    main()

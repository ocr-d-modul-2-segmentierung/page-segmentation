import argparse
import os

import tqdm

from pagesegmentation.lib.dataset import DatasetLoader
from pagesegmentation.lib.image_ops import fgpa_per_class, fgpa
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
    parser.add_argument("--calculate_fgpa", action="store_true",
                        help="output FgPA for each image (requires masks)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    dataset_loader = DatasetLoader(args.target_line_height, prediction=not args.calculate_fgpa)
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
    for i, pred in tqdm.tqdm(enumerate(predictor.predict(data))):
        if pred.data.mask is not None:
            total_fgpa = fgpa(pred.labels, pred.data.mask, pred.data.binary)
            fgpa_cls = fgpa_per_class(pred.labels, pred.data.mask, pred.data.binary, 2)

            print("total FgPA: {:.5}".format(total_fgpa))
            for cls, cls_fgpa in enumerate(fgpa_cls):
                print("class {} FgPA: {:.5}".format(cls, cls_fgpa))


if __name__ == "__main__":
    main()

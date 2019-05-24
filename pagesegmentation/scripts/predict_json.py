import argparse
import os

import tqdm
import numpy as np

from pagesegmentation.lib.dataset import DatasetLoader
from pagesegmentation.lib.image_ops import fgoverlap_per_class, fgpa
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
    parser.add_argument("--print_stats", nargs='?', default=None, const="yes",
                        help="output FgPA and overlap per class (`--print_stats all` for each image) (requires masks)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    dataset_loader = DatasetLoader(args.target_line_height, prediction=args.print_stats is None)
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
    size = len(data)
    avg_fgpa = 0.0
    overall_class_overlaps = np.zeros([3])
    classes_present = np.full([3], size)
    for i, pred in tqdm.tqdm(enumerate(predictor.predict(data))):
        if args.print_stats and pred.data.mask is not None:
            total_fgpa = fgpa(pred.labels, pred.data.mask, pred.data.binary)
            class_overlaps, tp, fp, fn = fgoverlap_per_class(pred.labels, pred.data.mask, pred.data.binary, 2)

            avg_fgpa += total_fgpa / size
            overall_class_overlaps += np.nan_to_num(class_overlaps)
            classes_present -= np.isnan(class_overlaps)

            if args.print_stats == "all":
                print("file {}".format(pred.data.image_path))
                print("FgPA: {:.5}".format(total_fgpa))
                for cls, (cls_fgpa, ctp, cfp, cfn) in enumerate(zip(class_overlaps, tp, fp, fn)):
                    print("class {} overlap: {:.5}, TP: {}, FP: {}, FN: {}".format(cls, cls_fgpa, ctp, cfp, cfn))

    nonexistent = classes_present == 0
    classes_present[nonexistent] = 1

    if args.print_stats:
        print("Average FgPA{:.5}".format(avg_fgpa))
        for cls, cls_fgpa in enumerate(overall_class_overlaps / classes_present):
            if nonexistent[cls]:
                print("class {} not in image".format(cls))
            else:
                print("class {} overlap: {:.5}".format(cls, cls_fgpa))


if __name__ == "__main__":
    main()

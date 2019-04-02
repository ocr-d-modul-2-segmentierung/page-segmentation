import argparse
import glob
import json
import os
from typing import Generator, Union, List

import tqdm

from pagesegmentation.lib.dataset import DatasetLoader, SingleData
from pagesegmentation.lib.predictor import Predictor, PredictSettings, Prediction


def glob_all(filenames):
    files = []
    for f in filenames:
        files += glob.glob(f)

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True,
                        help="Model to load")
    parser.add_argument("--char_height", type=int, required=False,
                        help="Average height of character m or n, ...")
    parser.add_argument("--target_line_height", type=int, default=6,
                        help="Scale the data images so that the line height matches this value (must be the same as in training)")
    parser.add_argument("--output", required=True,
                        help="Output dir")
    parser.add_argument("--binary", type=str, required=True, nargs="+",
                        help="directory name of the binary images")
    parser.add_argument("--images", type=str, required=True, nargs="+",
                        help="directory name of the images on which to train")
    parser.add_argument("--norm", type=str, required=False, nargs="+",
                        help="directory name of the norms on which to train")
    parser.add_argument("--keep_low_res", action="store_true",
                        help="keep low resolution prediction instead of rescaling output to orignal image size")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_file_paths = sorted(glob_all(args.images))
    binary_file_paths = sorted(glob_all(args.binary))
    if args.norm:
        norm_file_paths = sorted(glob_all(args.norm))
    else:
        norm_file_paths = []

    if len(image_file_paths) != len(binary_file_paths):
        raise Exception("Got {} images but {} binary images".format(len(image_file_paths), len(binary_file_paths)))

    print("Loading {} files with character height {}".format(len(image_file_paths), args.char_height))

    if not args.char_height and len(norm_file_paths) == 0:
        raise Exception("Either char height or norm files must be provided")

    if args.char_height:
        line_heights = [args.char_height] * len(image_file_paths)
    elif len(norm_file_paths) == 1:
        line_heights = [json.load(open(norm_file_paths[0]))["char_height"]] * len(image_file_paths)
    else:
        if len(norm_file_paths) != len(image_file_paths):
            raise Exception("Number of norm files must be one or equals the number of image files")
        line_heights = [json.load(open(n))["char_height"] for n in norm_file_paths]

    predictions = predict(args.output,
                          binary_file_paths,
                          image_file_paths,
                          line_heights,
                          target_line_height=args.target_line_height,
                          model=args.load,
                          high_res_output=not args.keep_low_res
                          )

    for i, pred in tqdm.tqdm(enumerate(predictions)):
        pass


def predict(output,
            binary_file_paths: List[str],
            image_file_paths: List[str],
            line_heights: Union[List[int], int],
            target_line_height: int,
            model: str,
            high_res_output: bool = True
            ) -> Generator[Prediction, None, None]:
    dataset_loader = DatasetLoader(target_line_height, prediction=True)

    if type(line_heights) is int:
        line_heights = [line_heights] * len(image_file_paths)

    data = dataset_loader.load_data(
        [SingleData(binary_path=b, image_path=i, line_height_px=n)
         for b, i, n in zip(binary_file_paths, image_file_paths, line_heights)]
    )

    settings = PredictSettings(
        mode='meta',
        network=os.path.abspath(model),
        output=output,
        high_res_output=high_res_output
    )
    predictor = Predictor(settings)

    return predictor.predict(data)


if __name__ == "__main__":
    main()

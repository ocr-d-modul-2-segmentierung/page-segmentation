import argparse
import os
import glob
from lib.dataset import DatasetLoader, SingleData, Dataset
import tqdm
import json
from lib.predictor import Predictor, PredictSettings


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

    mkdir(args.output)

    from lib.network import Network


    image_file_paths = sorted(glob_all(args.images))
    binary_file_paths = sorted(glob_all(args.binary))
    if args.norm:
        norm_file_paths = sorted(glob_all(args.norm))
    else:norm_file_paths = []

    if len(image_file_paths) != len(binary_file_paths):
        raise Exception("Got {} images but {} binary images".format(len(image_file_paths), len(binary_file_paths)))

    print("Loading {} files with character height {}".format(len(image_file_paths), args.char_height))

    if not args.char_height  and len(norm_file_paths) == 0:
        raise Exception("Either char height or norm files must be provided")


    dataset_loader = DatasetLoader(args.target_line_height, prediction=True)
    if args.char_height:
        data = dataset_loader.load_data(
            [SingleData(binary_path=b, image_path=i, line_height_px=args.char_height) for b, i in zip(binary_file_paths, image_file_paths)]
        )
    elif len(norm_file_paths) == 1:
        ch = json.load(open(norm_file_paths[0]))["char_height"]
        data = dataset_loader.load_data(
            [SingleData(binary_path=b, image_path=i, line_height_px=ch) for b, i in zip(binary_file_paths, image_file_paths)]
        )
    else:
        if len(norm_file_paths) != len(image_file_paths):
            raise Exception("Number of norm files must be one or equals the number of image files")

        data = dataset_loader.load_data(
            [SingleData(binary_path=b, image_path=i, line_height_px=json.load(open(n))["char_height"])
             for b, i, n in zip(binary_file_paths, image_file_paths, norm_file_paths)]
        )

    print("Creating net")
    settings = PredictSettings(
        mode='meta',
        network=os.path.abspath(args.load),
        output=args.output,
        high_res_output=not args.keep_low_res
    )
    predictor = Predictor(settings)

    print("Starting prediction")
    for i, pred in tqdm.tqdm(enumerate(predictor.predict(data))):
        pass


if __name__ == "__main__":
    main()
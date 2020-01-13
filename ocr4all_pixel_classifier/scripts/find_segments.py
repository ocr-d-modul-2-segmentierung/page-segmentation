import argparse
import os
import os.path

# remove when tensorflow#30559 is merged in 1.14.1
import warnings
from typing import Tuple, Optional, List, Callable

import cv2
import numpy as np

from ocr4all_pixel_classifier.lib.dataset import SingleData
from ocr4all_pixel_classifier.lib.pc_segmentation import find_segments
from ocr4all_pixel_classifier.lib.predictor import PredictSettings, Predictor, Masks
from ocr4all_pixel_classifier.scripts.compute_image_normalizations import compute_char_height

warnings.simplefilter(action='ignore', category=FutureWarning)

DEFAULT_IMAGE_MAP = {(255, 255, 255): [0, 'bg'],
                     (255, 0, 0): [1, 'text'],
                     (0, 255, 0): [2, 'image']}

DEFAULT_REVERSE_IMAGE_MAP = {v[1]: np.array(k) for k, v in DEFAULT_IMAGE_MAP.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--target-line-height", type=int, default=None,
                        help="Scale the data images so that the line height matches this value. If not specified, "
                             "try to determine automatically.")
    parser.add_argument("-I", "--image", type=str, default="./",
                        help="path to binary image to analyze")
    parser.add_argument("-O", "--output", type=str, default="./",
                        help="target directory for output")
    parser.add_argument("--load", type=str, default=None,
                        help="load an existing model")
    parser.add_argument("--gpu-allow-growth", action="store_true",
                        help="set allow_growth option for Tensorflow GPU. Use if getting CUDNN_INTERNAL_ERROR")
    args = parser.parse_args()

    image_dir, image_basename, image_ext = split_filename(args.image)

    process_dir = os.path.join(image_dir, image_basename)
    os.makedirs(process_dir, exist_ok=True)

    from shutil import copy
    copy(args.image, process_dir)  # TODO
    char_height = compute_char_height(args.image, True)

    image_map = DEFAULT_IMAGE_MAP
    rev_image_map = DEFAULT_REVERSE_IMAGE_MAP

    segmentation_dir = os.path.join(process_dir, "segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)

    resize_height = 300

    binary = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    orig_height, orig_width = binary.shape[0:2]

    from ocr4all_pixel_classifier.lib.dataset import prepare_images
    img, bin = prepare_images(binary, binary, args.target_line_height, char_height)

    masks = predict_masks(args.output,
                          img,
                          bin,
                          image_map,
                          char_height,
                          model=args.load,
                          post_processors=None,
                          gpu_allow_growth=args.gpu_allow_growth,
                          )

    image = masks.inverted_overlay
    height, width = image.shape[0:2]

    segments_text, segments_image = find_segments(orig_height, image, char_height, resize_height, rev_image_map)

    # TODO: write pagexml


def split_filename(image) -> Tuple[str, str, str]:
    image = os.path.basename(image)
    dir = os.path.dirname(image)
    base, ext = image.split(".", 1)
    return dir, base, ext


def predict_masks(output: Optional[str],
                  image: np.ndarray,
                  binary: np.ndarray,
                  color_map: dict,
                  line_height: int,
                  model: str,
                  post_processors: Optional[List[Callable[[np.ndarray, SingleData], np.ndarray]]] = None,
                  gpu_allow_growth: bool = False,
                  ) -> Masks:
    data = SingleData(binary=binary, image=image, original_shape=binary.shape, line_height_px=line_height)

    settings = PredictSettings(
        network=os.path.abspath(model),
        output=output,
        high_res_output=True,
        post_process=post_processors,
        color_map=color_map,
        n_classes=len(color_map),
        gpu_allow_growth=gpu_allow_growth,
    )
    predictor = Predictor(settings)

    return predictor.predict_masks(data)


if __name__ == "__main__":
    main()

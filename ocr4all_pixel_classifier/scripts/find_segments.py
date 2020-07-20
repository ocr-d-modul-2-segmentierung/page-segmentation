import argparse
import json
import os
import os.path

from typing import Tuple, Optional, List, Callable, Generator, Type, Any

import numpy as np
from dataclasses import dataclass

from pypagexml.ds import TextRegionTypeSub, CoordsTypeSub, ImageRegionTypeSub
from tqdm import tqdm

from ocr4all_pixel_classifier.lib.dataset import SingleData, color_to_label, label_to_colors
from ocr4all_pixel_classifier.lib.image_map import load_image_map_from_file, DEFAULT_IMAGE_MAP
from ocr4all_pixel_classifier.lib.pc_segmentation import find_segments, get_text_contours
from ocr4all_pixel_classifier.lib.predictor import PredictSettings, Predictor
from ocr4all_pixel_classifier.lib.output import Masks
from ocr4all_pixel_classifier.lib.util import glob_all, imread, imread_bin
from ocr4all_pixel_classifier.lib.xycut import render_regions, \
    render_morphological, render_xycut, AnyRegion
from ocr4all_pixel_classifier.lib.image_ops import compute_char_height


@dataclass
class SegmentationResult:
    text_segments: List[AnyRegion]
    image_segments: List[AnyRegion]
    original_shape: Tuple[int, int]
    path: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--target-line-height", type=int, default=6,
                        help="Scale the data images so that the line height matches this value.")

    parser.add_argument("-i", "--images", type=str, default=None, nargs="+",
                        help="path to images to analyze. if not specified, binaries are used.")
    parser.add_argument("-b", "--binary", type=str, default=None, nargs="+",
                        help="path to binary images to analyze")

    parser.add_argument("-m", "--method", type=str, choices=('xycut', 'morph'), default="xycut",
                        help="choose segmentation method")

    parser.add_argument("-n", "--norm", type=str, required=False, nargs="+",
                        help="use normalization files for input char height")
    parser.add_argument("--char_height", type=int, required=False,
                        help="Average height of character m or n in input")

    parser.add_argument("--resize-height", type=int, default=300,
                        help="Image scaling size to use for XYcut")
    parser.add_argument("-e", "--existing-preds-inverted", type=str, required=False, nargs="+",
                        help="Use given existing predictions (inverted overlay)")
    parser.add_argument("-c", "--existing-preds-color", type=str, required=False, nargs="+",
                        help="Use given existing predictions (color)")
    parser.add_argument("-o", "--xml-output-dir", type=str, default=None,
                        help="target directory for PageXML output")
    parser.add_argument("-O", "--render-output-dir", type=str, default=None,
                        help="target directory for rendered output")
    parser.add_argument("--load", "--model", type=str, default=None, dest="model",
                        help="load an existing model")
    parser.add_argument("--image-map", "--color_map", type=str, default=None, help="color_map to load",
                        dest="color_map")
    parser.add_argument("--gpu-allow-growth", action="store_true",
                        help="set allow_growth option for Tensorflow GPU. Use if getting CUDNN_INTERNAL_ERROR")
    parser.add_argument("--render", type=str, default=None,
                        choices=['png', 'dib', 'eps', 'gif', 'icns', 'ico', 'im', 'jpeg', 'msp',
                                 'pcx', 'ppm', 'sgi', 'tga', 'tiff', 'webp', 'xbm'],
                        help="load an existing model")
    args = parser.parse_args()

    image_map, rev_image_map = post_process_args(args, parser)

    if not args.existing_preds_inverted:
        results = predict_and_segment(args, image_map, rev_image_map)
    else:
        results = segment_existing(args, image_map, rev_image_map)

    for result in results:
        create_pagexml(result, args.xml_output_dir)
        if args.render:
            render_method = {
                'xycut': render_xycut,
                'morph': render_morphological
            }[args.method]

            render_regions(args.render_output_dir, args.render,
                           result.original_shape,
                           result.path,
                           rev_image_map,
                           render_method,
                           segments_text=result.text_segments,
                           segments_image=result.image_segments)


def predict_and_segment(args, image_map, rev_image_map) -> Generator[SegmentationResult, None, None]:
    def segment_new_predictions(binary_path, char_height, image_map, image_path, rev_image_map):
        masks = create_predictions(args.model, image_path, binary_path, char_height, args.target_line_height,
                                   args.gpu_allow_growth, image_map)
        overlay = masks.inverted_overlay
        if args.method == 'xycut':
            text, image = find_segments(overlay.shape[0], overlay, char_height, args.resize_height, rev_image_map)
        elif args.method == 'morph':
            text = get_text_contours(masks.fg_color_mask, char_height, rev_image_map)
            _, image = find_segments(masks.inverted_overlay.shape[0], masks.inverted_overlay, char_height,
                                     args.resize_height,
                                     rev_image_map, only_images=True)
        else:
            raise Exception("unknown method")

        return SegmentationResult(text, image, overlay.shape[0:2], image_path)

    results = (
        segment_new_predictions(binary_path, char_height, image_map, image_path, rev_image_map)
        for image_path, binary_path, char_height in
        tqdm(
            zip(args.image_paths, args.binary_paths, args.all_char_heights), unit='pages',
            total=len(args.image_paths)))
    return results


def segment_existing(args, image_map, rev_image_map) -> Generator[SegmentationResult, None, None]:
    def segment_existing_pred(binary_path, char_height, color_path, image_map, inverted_path, rev_image_map):
        overlay = imread(inverted_path)
        if args.method == 'xycut':
            text, image = find_segments(overlay.shape[0], overlay, char_height, args.resize_height, rev_image_map)
        elif args.method == 'morph':
            image, text = segment_existing_morph(binary_path, char_height, color_path, image_map, overlay,
                                                 rev_image_map)
        else:
            raise Exception("unknown method")

        return SegmentationResult(text, image, overlay.shape[0:2], inverted_path)

    def segment_existing_morph(binary_path, char_height, color_path, image_map, overlay, rev_image_map):
        binary = imread_bin(binary_path)
        color_mask = imread(color_path)
        label_mask = color_to_label(color_mask, image_map)
        label_mask[binary == 0] = 0
        fg_color_mask = label_to_colors(label_mask, image_map)
        text = get_text_contours(fg_color_mask, char_height, rev_image_map)
        _, image = find_segments(overlay.shape[0], overlay, char_height, args.resize_height,
                                 rev_image_map, only_images=True)
        return image, text

    results = (
        segment_existing_pred(binary_path, char_height, color_path, image_map, inverted_path, rev_image_map)
        for binary_path, inverted_path, color_path, char_height in
        tqdm(
            zip(args.binary_paths, args.existing_inverted_path, args.existing_color_path, args.all_char_heights),
            unit='pages', total=len(args.existing_inverted_path))
    )
    return results


def post_process_args(args, parser):
    if args.existing_preds_inverted and ((args.existing_preds_color and args.binary) or args.method == "xycut"):
        args.existing_inverted_path = sorted(glob_all(args.existing_preds_inverted))
        num_files = len(args.existing_inverted_path)
        if args.method == "morph":
            args.existing_color_path = sorted(glob_all(args.existing_preds_color))
            args.binary_paths = sorted(glob_all(args.binary))
        else:
            args.existing_color_path = [None] * num_files
            args.binary_paths = [None] * num_files
    elif args.method == "morph" \
            and (args.existing_preds_color or args.existing_preds_inverted) \
            and not (args.existing_preds_color and args.existing_preds_inverted and args.binary):
        return parser.error("Morphology method requires binaries and both existing predictions.\n"
                     "If you want to create new predictions, do not pass -e or -c.")
    elif args.binary:
        args.binary_paths = sorted(glob_all(args.binary))
        args.image_paths = sorted(glob_all(args.images)) if args.images else args.binary_paths
        num_files = len(args.binary_paths)
    elif args.method == "morph":
        return parser.error("Morphology method requires binary images.")
    else:
        return parser.error("Prediction requires binary images. Either supply binaries or existing preds")

    if not args.existing_preds_inverted and args.model is None:
        return parser.error("Prediction requires a model")

    if args.char_height:
        args.all_char_heights = [args.char_height] * num_files
    elif args.norm:
        norm_file_paths = sorted(glob_all(args.norm)) if args.norm else []
        if len(norm_file_paths) == 1:
            args.all_char_heights = [json.load(open(norm_file_paths[0]))["char_height"]] * num_files
        else:
            if len(norm_file_paths) != num_files:
                parser.error("Number of norm files must be one or equals the number of image files")
            args.all_char_heights = [json.load(open(n))["char_height"] for n in norm_file_paths]
    else:
        if not args.binary:
            parser.error("No binary files given, cannot auto-detect char height")
        args.all_char_heights = [compute_char_height(image, True)
                                 for image in tqdm(args.binary, desc="Auto-detecting char height", unit="pages")]

    if args.color_map:
        image_map = load_image_map_from_file(args.color_map)
    else:
        image_map = DEFAULT_IMAGE_MAP

    rev_image_map = {v[1]: np.array(k) for k, v in image_map.items()}

    return image_map, rev_image_map


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


def create_predictions(model, image_path, binary_path, char_height, target_line_height, gpu_allow_growth,
                       image_map=None):
    if image_map is None:
        image_map = DEFAULT_IMAGE_MAP

    image = imread(image_path)
    binary = imread_bin(binary_path)

    from ocr4all_pixel_classifier.lib.dataset import prepare_images
    img, bin = prepare_images(image, binary, target_line_height, char_height)

    return predict_masks(None,
                         img,
                         bin,
                         image_map,
                         char_height,
                         model=model,
                         post_processors=None,
                         gpu_allow_growth=gpu_allow_growth,
                         )


def create_pagexml(result: SegmentationResult, output_dir: Optional[str] = None):
    import pypagexml as pxml
    meta = pxml.ds.MetadataTypeSub(Creator="ocr4all_pixel_classifier", Created=pxml.ds.iso_now())
    doc = pxml.new_document_from_image(result.path, meta)

    def add_segment(segment: AnyRegion, index: int, idprefix: str, region_type: Type,
                    region_adder: Callable[[Any], None]):
        coords = segment.polygon_coords()
        id = f"{idprefix}{index}"
        region = region_type(id=id, Coords=CoordsTypeSub.with_points(coords))
        region_adder(region)

    for i, textseg in enumerate(result.text_segments):
        add_segment(textseg, i, "tr", TextRegionTypeSub, doc.get_Page().add_TextRegion)

    for i, imageseg in enumerate(result.image_segments):
        add_segment(imageseg, i, "ir", ImageRegionTypeSub, doc.get_Page().add_ImageRegion)

    if output_dir is None:
        output_file = os.path.splitext(result.path)[0] + ".xml"
    else:
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(output_dir))[0] + ".xml")
    doc.saveAs(result.path + ".xml", level=0)


if __name__ == "__main__":
    main()

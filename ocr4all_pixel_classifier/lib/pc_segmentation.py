#!/usr/bin/env python3

import argparse
import datetime
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List, Union

import cv2
import numpy as np
from dataclasses import dataclass

ColorMapping = Dict[str, np.ndarray]


@dataclass
class Segment:
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    done: bool = False

    def of(self, image: np.ndarray):
        return image[self.y_start:self.y_end, self.x_start:self.x_end]


@dataclass
class Segment1D:
    start: int
    end: int


@dataclass
class Gap:
    start: int
    length: int


DEFAULT_COLOR_MAPPING = {
    "image": np.array([0, 255, 0]),
    "text": np.array([0, 0, 255]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inverted", type=str, help="Path to the inverted image which should be analyzed",
                        required=True)
    parser.add_argument("--binary", type=str, help="Path to the binary image", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to write segmentation files", required=True)
    parser.add_argument("--char_height", type=int, help="Average height of character m or n, ...", required=True)
    args = parser.parse_args()

    pc_segment_main(args.binary, args.inverted, args.output_dir, args.char_height)


def pc_segment_main(orig_image: str, inverted_image: str, char_height: int, output_dir: str,
                    write_box_images: bool = False):
    if not os.path.isfile(inverted_image):
        print("Error: Inverted image file not found: " + inverted_image, file=sys.stderr)
        raise SystemExit()
    if not os.path.isfile(orig_image):
        print("Error: Binary image file not found", file=sys.stderr)
        raise SystemExit()

    # Get image name
    from ocr4all_pixel_classifier.scripts.find_segments import split_filename
    _, image_basename, image_ext = split_filename(orig_image)
    if image_basename == "" or image_ext == "":
        print("Error: New image name could not be determined", file=sys.stderr)
        raise SystemExit()

    # Load and get dimensions of the binary image
    binary = cv2.imread(orig_image)

    # Load and get dimensions of the inverted image
    image = cv2.imread(inverted_image)

    # Determines the height in px to which the image should be resized before analyzing
    # Speedup vs Accuracy ?
    resize_height = 300

    # Color mapping to match results of the Pixel Classifier
    color_mapping = DEFAULT_COLOR_MAPPING

    orig_height, orig_width = binary.shape[0:2]

    segments_text, segments_image = find_segments(orig_height, image, char_height, resize_height, color_mapping)

    if write_box_images:
        output_name = image_basename + "_cut" + image_ext
        write_images(image, output_dir, output_name, segments_image, segments_text, color_mapping)

    # Write PageXML
    create_page_xml(orig_image, orig_width, orig_height, resize_height, segments_text, segments_image,
                    os.path.join(output_dir, "clip_" + image_basename + image_ext + ".xml"))

    # Create an image for each text segment for OCR
    create_segment_images(binary, image_basename, image_ext, resize_height, segments_text, output_dir)


def find_segments(orig_height: int, image: np.ndarray, char_height: int, resize_height: int,
                  color_mapping: ColorMapping) -> Tuple[List[Segment], List[Segment]]:
    # Scale image to specific height for more generic usage of threshold values
    scale_percent = resize_height / image.shape[0]
    height = resize_height
    width = int(image.shape[1] * scale_percent)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    # Get resizing factor of the scaled image compared to the original one (binary)
    absolute_resize_factor = height / orig_height

    # Determines how many pixels in one line/column need to exist to indicate a match
    px_threshold_line = int(char_height * absolute_resize_factor)
    px_threshold_column = int(char_height * absolute_resize_factor)
    # Determines the size of a gap in pixels to split the found matches into segments
    split_size_horizontal = int(char_height * 2 * absolute_resize_factor)
    split_size_vertical = int(char_height * absolute_resize_factor)

    # Calculate x-y-cut and get its segments
    segments_text = get_xy_cut(image, px_threshold_line, px_threshold_column, split_size_horizontal,
                               split_size_vertical, color_mapping["text"])
    segments_image = get_xy_cut(image, px_threshold_line, px_threshold_column, split_size_horizontal,
                                split_size_vertical, color_mapping["image"])

    return segments_text, segments_image


def write_images(image: np.ndarray, output_dir: str, output_name: str,
                 segments_image: List[Segment], segments_text: List[Segment],
                 color_mapping: ColorMapping) -> None:
    # Mark the found segments in the image
    for segment in segments_text:
        cv2.rectangle(image, (segment.x_start, segment.y_start), (segment.x_end, segment.y_end),
                      color_mapping["text"], 1)
    for segment in segments_image:
        cv2.rectangle(image, (segment.x_start, segment.y_start), (segment.x_end, segment.y_end),
                      color_mapping["image"], 1)
    cv2.imwrite(os.path.join(output_dir, output_name), image)


def get_indication(image: np.ndarray, segment: Segment, color_match: Union[int, np.ndarray],
                   threshold: int, axis: int) -> np.ndarray:
    mask = segment.of(image) == color_match
    if len(image.shape) > 2:
        mask = mask.all(axis=-1)
    return np.count_nonzero(mask, axis=axis) >= threshold


def consecutive(data: np.ndarray, stepsize=1) -> List[np.ndarray]:
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def get_gaps(indication: np.ndarray) -> List[Gap]:
    return [Gap(start=x[0], length=len(x)) for x in consecutive(np.where(~indication)[0])]


def get_sliced_coords(gaps: List[Gap], default_start: int, default_end: int) -> Tuple[int, int]:
    sliced_start = gaps[0].length if gaps[0].start == 0 else default_start
    sliced_end = gaps[-1].start if (gaps[-1].start + gaps[-1].length) >= default_end else default_end

    return sliced_start, sliced_end


def get_segments(gaps: List[Gap], start_px: int, end_px: int, px_threshold, split_size) -> List[Segment1D]:
    segments = []
    segment_start = start_px

    for gap in gaps:
        if gap.start > start_px and (gap.start + gap.length) < end_px:
            if gap.length > split_size:
                if (segment_start + gap.start) > px_threshold:
                    segments.append(Segment1D(start=segment_start, end=gap.start))
                    segment_start = gap.start + gap.length

    # Always append last segment that remains
    if segment_start < end_px and (segment_start + end_px) > px_threshold:
        segments.append(Segment1D(start=segment_start, end=end_px))

    return segments


def single_cut(image: np.ndarray,
               segment: Segment,
               px_threshold: int, split_size: int,
               color_match: int,
               axis: int) -> List[Segment1D]:
    indication = get_indication(image, segment, color_match, px_threshold, axis=axis)
    gaps = get_gaps(indication)
    default_start, default_end = (segment.y_start, segment.y_end) if axis == 1 else (segment.x_start, segment.x_end)
    sliced_start, sliced_end = get_sliced_coords(gaps, default_start, default_end)
    return get_segments(gaps, sliced_start, sliced_end, px_threshold, split_size)


def get_xy_cut(image: np.ndarray,
               px_threshold_line: int, px_threshold_column: int,
               split_size_horizontal: int, split_size_vertical: int,
               color_match: Union[int, np.ndarray]) -> List[Segment]:
    height, width = image.shape[0:2]
    segments_final = []
    segments_new = [Segment(done=False, y_start=0, y_end=height, x_start=0, x_end=width)]

    while len(segments_final) != len(segments_new):
        segments_final = segments_new
        segments_new = []

        for segment in segments_final:
            # Segment is in final cut and cannot be splitted further
            if segment.done:
                segments_new.append(segment)
                continue

            y_segments = single_cut(image, segment, px_threshold_line, split_size_horizontal, color_match, axis=1)
            x_segments = single_cut(image, segment, px_threshold_column, split_size_vertical, color_match, axis=0)
            segments_cut = [Segment(done=False,
                                    y_start=y_segment.start, y_end=y_segment.end,
                                    x_start=x_segment.start, x_end=x_segment.end
                                    )
                            for y_segment in y_segments
                            for x_segment in x_segments]

            if len(segments_cut) == 1:
                # Segment stayed the same (maybe sliced), make it final
                segments_cut[0].done = True
                segments_new.append(segments_cut[0])
            else:
                # Segment is splitted again in smaller pieces
                # Loop will be rerun to verify new segments
                segments_new = segments_new + segments_cut

    return segments_new


def get_xml_point_string(segment: Segment, coord_factor: float):
    return " ".join(["{:.0f},{:.0f}"] * 4).format(*[x * coord_factor for x in [
        segment.x_start, segment.y_start,
        segment.x_end, segment.y_start,
        segment.x_end, segment.y_end,
        segment.x_start, segment.y_end,
    ]])


def create_page_xml(filename: str, width: int, height: int, resize_height: int,
                    segments_text: List[Segment], segments_image: List[Segment],
                    outfile: str):
    coord_factor = height / resize_height

    pcgts = ET.Element("PcGts", {
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15/pagecontent.xsd"
    })

    metadata = ET.SubElement(pcgts, "Metadata")
    creator = ET.SubElement(metadata, "Creator")
    creator.text = "User123"
    created = ET.SubElement(metadata, "Created")
    created.text = datetime.datetime.now().isoformat()
    last_change = ET.SubElement(metadata, "LastChange")
    last_change.text = datetime.datetime.now().isoformat()

    page = ET.SubElement(pcgts, "Page", {
        "imageFilename": filename,
        "imageWidth": str(width),
        "imageHeight": str(height),
    })

    count = 0
    for segment in segments_text:
        region = ET.SubElement(page, "TextRegion", {"id": "r" + str(count), "type": "paragraph"})
        coords = ET.SubElement(region, "Coords", {"points": get_xml_point_string(segment, coord_factor)})
        count += 1
    for segment in segments_image:
        region = ET.SubElement(page, "GraphicRegion", {"id": "r" + str(count)})
        coords = ET.SubElement(region, "Coords", {"points": get_xml_point_string(segment, coord_factor)})
        count += 1

    tree = ET.ElementTree(pcgts)
    tree.write(outfile, xml_declaration=True, encoding='utf-8', method="xml")


def create_segment_images(image: np.ndarray, image_basename: str, image_ext: str, resize_height: int,
                          segments: List[Segment],
                          output_dir: str) -> None:
    coord_factor = image.shape[0] / resize_height

    count = 0
    for segment in segments:
        segment_image = image[
                        int(segment.y_start * coord_factor):int(segment.y_end * coord_factor),
                        int(segment.x_start * coord_factor):int(segment.x_end * coord_factor)
                        ]
        cv2.imwrite(os.path.join(output_dir, image_basename + "__" + '{:03}'.format(count) + "__paragraph" + image_ext),
                    segment_image)
        count += 1


if __name__ == "__main__":
    main()

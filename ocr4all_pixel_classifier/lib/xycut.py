from typing import Union, List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw

from dataclasses import dataclass

RGBColor = Tuple[int, int, int]


@dataclass
class Segment:
    x_start: int
    y_start: int
    x_end: int
    y_end: int

    def of(self, image: np.ndarray):
        return image[self.y_start:self.y_end, self.x_start:self.x_end]

    def scale(self, factor: float):
        return Segment(
            x_start=int(self.x_start * factor),
            y_start=int(self.y_start * factor),
            x_end=int(self.x_end * factor),
            y_end=int(self.y_end * factor),
        )

    def as_xy(self) -> List[Tuple[int, int]]:
        return [(self.y_start, self.x_start), (self.y_end, self.x_end)]

    def render(self, canvas: ImageDraw, color: RGBColor):
        canvas.rectangle(self.as_xy(), fill=color, outline=color)


def render_all(size: Tuple[int, int], segment_groups: List[Tuple[RGBColor, List[Segment]]],
               base_color: Tuple[int, int, int] = (255, 255, 255)) -> Image:
    pil_image = Image.new('RGB', size, base_color)
    canvas = ImageDraw.Draw(pil_image)
    for color, segments in segment_groups:
        for seg in segments:
            seg.render(canvas, color)
    return pil_image


@dataclass
class Segment1D:
    start: int
    end: int

    def __len__(self):
        return self.end - self.start


@dataclass
class Gap:
    start: int
    length: int


def single_color(image: np.ndarray, color: Union[int, np.ndarray]):
    mask = image == color
    if len(image.shape) > 2:
        mask = mask.all(axis=-1)
    return mask


def do_xy_cut(image: np.ndarray,
              px_threshold_line: int, px_threshold_column: int,
              split_size_horizontal: int, split_size_vertical: int,
              color_match: Union[int, np.ndarray]) -> List[Segment]:
    mask = single_color(image, color_match)
    return recursive_cut(mask,
                         (px_threshold_line, px_threshold_column),
                         (split_size_horizontal, split_size_vertical),
                         axis=0, orig_image=mask
                         )


def consecutive(data: np.ndarray, stepsize=1) -> List[np.ndarray]:
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def get_gaps(indication: np.ndarray) -> List[Gap]:
    # return [Gap(start=x[0], length=len(x)) for x in consecutive(np.where(~indication)[0]) if x is not []]
    no_indication = np.where(~indication)
    cons = consecutive(no_indication[0])
    return [Gap(start=x[0], length=len(x)) for x in cons if len(x) > 0]


def relative_seg(shape, start, end, axis, pos):
    if axis == 0:
        return Segment(x_start=pos[1] + start,
                       x_end=pos[1] + end,
                       y_start=pos[0],
                       y_end=pos[0] + shape[1])
    else:
        return Segment(x_start=pos[1],
                       x_end=pos[1] + shape[0],
                       y_start=pos[0] + start,
                       y_end=pos[0] + end)


def recursive_cut(image: np.ndarray,
                  threshold: Tuple[int, int],
                  split_size: Tuple[int, int],
                  axis: int = 0,
                  position: Tuple[int, int] = (0, 0),
                  orig_image=None,
                  end_recurse=False
                  ) -> List[Segment]:
    white_lines = free_indices(image, threshold[axis], axis)
    gaps = get_gaps(white_lines)
    if len(gaps) == 0:
        return [relative_seg(image.shape, 0, image.shape[axis], axis, position)]

    segments_for_axis = get_segments(gaps, image.shape[axis], threshold[axis], split_size[axis])

    if end_recurse:
        return [relative_seg(image.shape, s.start, s.end, axis, position) for s in segments_for_axis]

    recursive_segments = []

    for seg in segments_for_axis:
        if len(seg) > threshold[axis]:
            if axis == 1:
                slice = image[seg.start:seg.end, :]
                pos = (position[0], position[1] + seg.start)
            else:
                slice = image[:, seg.start:seg.end]
                pos = (position[0] + seg.start, position[1])

            recursive_segments += recursive_cut(slice, threshold, split_size, 1 - axis, pos, orig_image,
                                                len(segments_for_axis) == 1)

    return recursive_segments


def get_segments(gaps: List[Gap], length: int, px_threshold, split_size) -> List[Segment1D]:
    # remove small gaps
    gaps = [Gap(0, 0)] + [g for g in gaps if g.length >= split_size] + [Gap(length, 0)]

    segments = []
    for gap, nextgap in zip(gaps, gaps[1:]):
        if nextgap.start - (gap.start + gap.length) > px_threshold:
            segments.append(Segment1D(gap.start + gap.length, nextgap.start))

    return segments


def free_indices(image: np.ndarray, threshold: int, axis: int) -> np.ndarray:
    return np.count_nonzero(image, axis=axis) >= threshold

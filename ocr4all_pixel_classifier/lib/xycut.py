from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Tuple, TypeVar

import numpy as np

RGBColor = Tuple[int, int, int]


class Region(ABC):
    @abstractmethod
    def polygon_coords(self) -> Union[List[Tuple[int, int]], np.ndarray]:
        pass

    @abstractmethod
    def scale(self, factor: float) -> 'Region':
        pass


@dataclass
class CVContour(Region):
    contour: np.ndarray

    def __post_init__(self):
        self.contour = np.squeeze(self.contour)

    def polygon_coords(self) -> Union[List[Tuple[int, int]], np.ndarray]:
        return np.squeeze(self.contour)

    def scale(self, factor: float) -> 'CVContour':
        return CVContour((self.contour * factor).astype('int32'))


@dataclass
class RectSegment(Region):
    x_start: int
    y_start: int
    x_end: int
    y_end: int

    def of(self, image: np.ndarray):
        return image[self.y_start:self.y_end, self.x_start:self.x_end]

    def scale(self, factor: float) -> 'RectSegment':
        return RectSegment(
            x_start=int(self.x_start * factor),
            y_start=int(self.y_start * factor),
            x_end=int(self.x_end * factor),
            y_end=int(self.y_end * factor),
        )

    def as_xy(self) -> List[Tuple[int, int]]:
        return [(self.y_start, self.x_start), (self.y_end, self.x_end)]

    def polygon_coords(self) -> Union[List[Tuple[int, int]], np.ndarray]:
        #
        # (x_start, y_start) 1--------2 (x_end, y_start)
        #                    |        |
        #                    |        |
        #   (x_start, y_end) 4--------3 (x_end, y_end)
        #
        return [
            (self.x_start, self.y_start),
            (self.x_end, self.y_start),
            (self.x_end, self.y_end),
            (self.x_start, self.y_end),
        ]


AnyRegion = TypeVar('AnyRegion', Region, RectSegment, CVContour)


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


def do_xy_cut(binary_image: np.ndarray, px_threshold_line: int, px_threshold_column: int,
              split_size_horizontal: int, split_size_vertical: int) -> List[RectSegment]:
    """
    Runs an xy cut algorithm on an image to find rectangular regions.
    :param binary_image: boolean np array, true resp. 1 is foreground
    :param px_threshold_line: minimum height required to further split a region horizontally.
    :param px_threshold_column:  minimum width required to further split a region vertically.
    :param split_size_horizontal: pixels of free space required for a horizontal cut.
    :param split_size_vertical:  pixels of free space required for a vertical cut.
    :return:
    """
    return recursive_cut(binary_image,
                         (px_threshold_line, px_threshold_column),
                         (split_size_horizontal, split_size_vertical),
                         axis=0)


def _get_gaps(indication: np.ndarray) -> List[Gap]:
    # return [Gap(start=x[0], length=len(x)) for x in consecutive(np.where(~indication)[0]) if x is not []]
    no_indication = np.where(~indication)
    data = no_indication[0]
    consecutive = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
    return [Gap(start=x[0], length=len(x)) for x in consecutive if len(x) > 0]


def _relative_seg(shape, start, end, axis, pos):
    if axis == 0:
        return RectSegment(x_start=pos[1] + start,
                           x_end=pos[1] + end,
                           y_start=pos[0],
                           y_end=pos[0] + shape[1])
    else:
        return RectSegment(x_start=pos[1],
                           x_end=pos[1] + shape[0],
                           y_start=pos[0] + start,
                           y_end=pos[0] + end)


def recursive_cut(image: np.ndarray,
                  threshold: Tuple[int, int],
                  split_size: Tuple[int, int],
                  axis: int = 0,
                  position: Tuple[int, int] = (0, 0),
                  end_recurse=False
                  ) -> List[RectSegment]:

    threshold1 = threshold[axis]
    white_lines = np.count_nonzero(image, axis=axis) >= threshold1
    gaps = _get_gaps(white_lines)
    if len(gaps) == 0:
        return [_relative_seg(image.shape, 0, image.shape[axis], axis, position)]

    segments_for_axis = _get_segments(gaps, image.shape[axis], threshold[axis], split_size[axis])

    if end_recurse:
        return [_relative_seg(image.shape, s.start, s.end, axis, position) for s in segments_for_axis]

    recursive_segments = []

    for seg in segments_for_axis:
        if len(seg) > threshold[axis]:
            if axis == 1:
                slice = image[seg.start:seg.end, :]
                pos = (position[0], position[1] + seg.start)
            else:
                slice = image[:, seg.start:seg.end]
                pos = (position[0] + seg.start, position[1])

            recursive_segments += recursive_cut(slice, threshold, split_size, 1 - axis, pos,
                                                len(segments_for_axis) == 1)

    return recursive_segments


def _get_segments(gaps: List[Gap], length: int, px_threshold, split_size) -> List[Segment1D]:
    # remove small gaps
    gaps = [Gap(0, 0)] + [g for g in gaps if g.length >= split_size] + [Gap(length, 0)]

    segments = []
    for gap, nextgap in zip(gaps, gaps[1:]):
        if nextgap.start - (gap.start + gap.length) > px_threshold:
            segments.append(Segment1D(gap.start + gap.length, nextgap.start))

    return segments



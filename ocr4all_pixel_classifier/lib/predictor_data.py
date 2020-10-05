# this file is separated from predictor.py to prevent tensorflow from loading before it is needed

from dataclasses import dataclass
from typing import NamedTuple, Optional, List, Callable

import numpy as np
from ocr4all.colors import ColorMap

from ocr4all_pixel_classifier.lib.dataset import SingleData


class Prediction(NamedTuple):
    labels: np.ndarray
    probabilities: np.ndarray
    data: SingleData


@dataclass
class PredictSettings:
    network: str = None
    output: str = None
    high_res_output: bool = False
    color_map: Optional[ColorMap] = None  # Only needed for generating colored images
    n_classes: int = -1
    post_process: Optional[List[Callable[[np.ndarray, SingleData], np.ndarray]]] = None
    gpu_allow_growth: bool = False
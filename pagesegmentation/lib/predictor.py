import os
from typing import NamedTuple, Generator, List, Callable, Optional

import numpy as np
import scipy.misc as misc
import skimage.io as img_io
import tensorflow as tf
from dataclasses import dataclass
from tqdm import tqdm

from .dataset import Dataset, SingleData
from .model import model as default_model
from .network import Network


class Prediction(NamedTuple):
    labels: np.ndarray
    probabilities: np.ndarray
    data: SingleData


@dataclass
class PredictSettings:
    n_classes: int = -1  # if mode == meta this is not required
    network: str = None
    output: str = None
    mode: str = 'meta'  # meta, deploy or test
    high_res_output: bool = False
    post_process: Optional[List[Callable[[np.ndarray, SingleData], np.ndarray]]] = None


class Predictor:
    def __init__(self, settings: PredictSettings, network: Network = None, model=default_model):
        self.settings = settings

        self.network = network
        if not network:
            graph = tf.Graph()
            session = tf.Session(graph=graph)
            if settings.mode == 'meta':
                self.network = Network('meta', graph, session, model, settings.n_classes, 0, meta=settings.network)
            elif settings.mode == 'deploy':
                self.network = Network(settings.mode, graph, session, model, settings.n_classes, 0)
                self.network.load_weights(settings.network, restore_only_trainable=True)
            else:
                raise Exception('Invalid value, either meta or deploy')

        if settings.output:
            output_dir = settings.output
            os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "color"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "inverted"), exist_ok=True)

    def predict(self, dataset: Dataset) -> Generator[Prediction, None, None]:
        for data in dataset.data:
            pred, prob = self.network.predict_single_data(data)

            if self.settings.post_process:
                for processor in self.settings.post_process:
                    pred = processor(pred, data)

            self.output_data(pred, data)
            yield Prediction(pred, prob, data)

    def test(self, dataset: Dataset, n_classes: int = 0):
        """
        Run classifier on given dataset and evaluate
        :param dataset: input data
        :param n_classes: if nonzero: calculate FgPA per class for classes up to n_classes
        :return: accuracy, FgPA, array with FgPA for class i at index i
        """
        self.network.set_data(dataset)
        total_a, total_fg = 0, 0
        total_fg_per_class = np.zeros(n_classes)
        for pred, a, fg, data in tqdm(self.network.test_dataset(), total=self.network.n_data()):
            total_a += a / self.network.n_data()
            total_fg += fg / self.network.n_data()
            if n_classes > 0:
                total_fg_per_class += fgpa_per_class(pred, data.mask, data.binary, n_classes) / self.network.n_data()

            self.output_data(pred, data)

        return total_a, total_fg, total_fg_per_class

    def output_data(self, pred, data: SingleData):
        if len(pred.shape) == 3:
            assert (pred.shape[0] == 1)
            pred = pred[0]

        if self.settings.output:
            from .dataset import label_to_colors
            if data.output_path:
                filename = data.output_path
                dir = os.path.dirname(filename)
                if dir:
                    os.makedirs(dir, exist_ok=True)
            else:
                filename = os.path.basename(data.image_path)

            color_mask = label_to_colors(pred)
            foreground = np.stack([(1 - data.image / 255)] * 3, axis=-1)
            inv_binary = data.binary

            if self.settings.high_res_output:
                color_mask = misc.imresize(color_mask[data.xpad:, data.ypad:], data.original_shape, interp="nearest")
                foreground = misc.imresize(foreground[data.xpad:, data.ypad:], data.original_shape) / 255
                inv_binary = misc.imresize(inv_binary[data.xpad:, data.ypad:], data.original_shape, interp="nearest")

            inv_binary = np.stack([inv_binary] * 3, axis=-1)
            overlay_mask = np.ndarray.astype(color_mask * foreground, dtype=np.uint8)
            inverted_overlay_mask = np.ndarray.astype(color_mask * inv_binary, dtype=np.uint8)

            img_io.imsave(os.path.join(self.settings.output, "color", filename), color_mask)
            img_io.imsave(os.path.join(self.settings.output, "overlay", filename), overlay_mask)
            img_io.imsave(os.path.join(self.settings.output, "inverted", filename), inverted_overlay_mask)

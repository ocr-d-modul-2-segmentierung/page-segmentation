from typing import NamedTuple, Generator
from .network import Network
import tensorflow as tf
from .model import model as default_model
from .dataset import Dataset, SingleData
from tqdm import tqdm
import skimage.io as img_io
import numpy as np
import os
from dataclasses import dataclass
import scipy.misc as misc


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Prediction(NamedTuple):
    labels: np.ndarray
    probabilities: np.ndarray
    data: SingleData


@dataclass
class PredictSettings:
    n_classes: int = -1 # if mode == meta this is not required
    network: str = None
    output: str = None
    mode: str = 'meta'  # meta, deploy or test
    high_res_output: bool = False


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
            mkdir(os.path.join(output_dir, "overlay"))
            mkdir(os.path.join(output_dir, "color"))
            mkdir(os.path.join(output_dir, "inverted"))

    def predict(self, dataset: Dataset) -> Generator[Prediction, None, None]:
        for data in dataset.data:
            pred, prob = self.network.predict_single_data(data)
            self.output_data(pred, data)
            yield Prediction(pred, prob, data)

    def test(self, dataset: Dataset):
        self.network.set_data(dataset)
        total_a, total_fg = 0, 0
        for pred, a, fg, data in tqdm(self.network.test_dataset(), total=self.network.n_data()):
            total_a += a / self.network.n_data()
            total_fg += fg / self.network.n_data()

            self.output_data(pred, data)

        return total_a, total_fg

    def output_data(self, pred, data: SingleData):
        if len(pred.shape) == 3:
            assert(pred.shape[0] == 1)
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
            inv_binary = np.stack([data.binary] * 3, axis=-1)

            if self.settings.high_res_output:
                color_mask = misc.imresize(color_mask[data.xpad:, data.ypad:], data.original_shape, interp="nearest")
                foreground = misc.imresize(foreground[data.xpad:, data.ypad:], data.original_shape)
                inv_binary = misc.imresize(inv_binary[data.xpad:, data.ypad:], data.original_shape, interp="nearest")

            overlay_mask = np.ndarray.astype(color_mask * foreground, dtype=np.uint8)
            inverted_overlay_mask = np.ndarray.astype(color_mask * inv_binary, dtype=np.uint8)

            img_io.imsave(os.path.join(self.settings.output, "color", filename), color_mask)
            img_io.imsave(os.path.join(self.settings.output, "overlay", filename), overlay_mask)
            img_io.imsave(os.path.join(self.settings.output, "inverted", filename), inverted_overlay_mask)

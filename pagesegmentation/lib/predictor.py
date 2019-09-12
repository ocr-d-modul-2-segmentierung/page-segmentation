import os
from typing import NamedTuple, Generator, List, Callable, Optional

import numpy as np
import skimage.io as img_io
from skimage.transform import resize
from dataclasses import dataclass
from tqdm import tqdm

from pagesegmentation.lib.dataset import Dataset, SingleData
from pagesegmentation.lib.network import Network


class Prediction(NamedTuple):
    labels: np.ndarray
    probabilities: np.ndarray
    data: SingleData


@dataclass
class PredictSettings:
    network: str = None
    output: str = None
    high_res_output: bool = False
    color_map: dict = None # Only needed for generating colored images
    post_process: Optional[List[Callable[[np.ndarray, SingleData], np.ndarray]]] = None


class Predictor:
    def __init__(self, settings: PredictSettings, network: Network = None):
        self.settings = settings
        self.network = network

        if not network:
            self.network = Network("Predict", model=os.path.abspath(self.settings.network))
        if settings.output:
            output_dir = settings.output
            os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "color"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "inverted"), exist_ok=True)

    def predict(self, dataset: Dataset) -> Generator[Prediction, None, None]:
        for data in dataset.data:
            logit, prob, pred = self.network.predict_single_data(data)

            if self.settings.post_process:
                for processor in self.settings.post_process:
                    pred = processor(pred, data)

            self.output_data(pred, data)
            yield Prediction(pred, prob, data)

    def output_data(self, pred, data: SingleData):
        if len(pred.shape) == 3:
            assert (pred.shape[0] == 1)
            pred = pred[0]

        if self.settings.output:
            from .dataset import label_to_colors
            if data.output_path:
                filename = data.output_path
                dir = os.path.dirname(filename)
                if os.path.isabs(dir):
                    os.makedirs(dir, exist_ok=True)
                elif dir:
                    for category in ["color", "overlay", "inverted"]:
                        os.makedirs(os.path.join(self.settings.output, category, dir), exist_ok=True)
            else:
                filename = os.path.basename(data.image_path)

            color_mask = label_to_colors(pred, colormap=self.settings.color_map)
            foreground = np.stack([(1 - data.binary)] * 3, axis=-1)
            inv_binary = data.binary

            if self.settings.high_res_output:
                color_mask = resize(color_mask, data.original_shape, order=0)
                foreground = resize(foreground, data.original_shape) / 255
                inv_binary = resize(inv_binary, data.original_shape, order=0)
            inv_binary = np.stack([inv_binary] * 3, axis=-1)
            overlay_mask = color_mask.copy()
            overlay_mask[foreground == 0] = 0
            inverted_overlay_mask = color_mask.copy()
            inverted_overlay_mask[inv_binary == 0] = 0

            img_io.imsave(os.path.join(self.settings.output, "color", filename), color_mask)
            img_io.imsave(os.path.join(self.settings.output, "overlay", filename), overlay_mask)
            img_io.imsave(os.path.join(self.settings.output, "inverted", filename), inverted_overlay_mask)

if __name__ == "__main__":
    from pagesegmentation.lib.dataset import DatasetLoader
    from pagesegmentation.scripts.generate_image_map import load_image_map_from_file
    import os
    dataset_dir = '/home/alexander/Dokumente/virutal_stafflines/'
    image_map = load_image_map_from_file(os.path.join(dataset_dir, 'image_map.json'))
    dataset_loader = DatasetLoader(8, color_map=image_map)
    train_data = dataset_loader.load_data_from_json(
        [os.path.join(dataset_dir, 't.json')], "train")
    test_data = dataset_loader.load_data_from_json(
        [os.path.join(dataset_dir, 't.json')], "test")
    eval_data = dataset_loader.load_data_from_json(
        [os.path.join(dataset_dir, 't.json')], "eval")
    settings = PredictSettings(network='/home/alexander/Dokumente/virutal_stafflines/model.h5')
    predictor = Predictor(settings)
    for x in predictor.predict(test_data):
        print(x)

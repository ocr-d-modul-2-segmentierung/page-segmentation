import os
from typing import Generator

from ocr4all_pixel_classifier.lib.predictor_data import Prediction, PredictSettings
from ocr4all_pixel_classifier.lib.dataset import Dataset, SingleData
from ocr4all_pixel_classifier.lib.network import Network, tf_backend_allow_growth
from ocr4all_pixel_classifier.lib.output import Masks, scale_to_original_shape, generate_output_masks


class Predictor:
    def __init__(self, settings: PredictSettings, network: Network = None):
        self.settings = settings
        self.network = network

        if settings.gpu_allow_growth:
            tf_backend_allow_growth()

        if not network:
            self.network = Network("Predict", n_classes=settings.n_classes,
                                   model=os.path.abspath(self.settings.network))
        if settings.output:
            output_dir = settings.output
            os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "color"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "inverted"), exist_ok=True)

    def predict(self, dataset: Dataset) -> Generator[Prediction, None, None]:
        for data in dataset.data:
            prediction = self.predict_single(data)
            yield prediction

    def predict_single(self, data: SingleData) -> Prediction:
        logit, prob, pred = self.network.predict_single_data(data)

        if self.settings.high_res_output:
            data, pred = scale_to_original_shape(data, pred)

        if self.settings.post_process:
            for processor in self.settings.post_process:
                pred = processor(pred, data)

        return Prediction(pred, prob, data)

    def predict_masks(self, data: SingleData) -> Masks:
        logit, prob, pred = self.network.predict_single_data(data)

        if self.settings.high_res_output:
            data, pred = scale_to_original_shape(data, pred)

        if self.settings.post_process:
            for processor in self.settings.post_process:
                pred = processor(pred, data)

        return generate_output_masks(data, pred, self.settings.color_map)



from ocr4all_pixel_classifier.lib.dataset import Dataset
from ocr4all_pixel_classifier.lib.callback import TrainProgressCallback
from typing import NamedTuple, Optional, List
from ocr4all_pixel_classifier.lib.metrics import Loss, Monitor
from ocr4all_pixel_classifier.lib.model import Architecture, Optimizers
import numpy as np
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class AugmentationSettings(NamedTuple):
    rotation_range: float = 2.5
    width_shift_range: float = 0.025
    height_shift_range: float = 0.025
    shear_range: float = 0.00
    zoom_range: List[float] = [0.95, 1.05]
    horizontal_flip: bool = False
    vertical_flip: bool = False
    brightness_range: Optional[List[float]] = None

    image_fill_mode: str = 'nearest'
    binary_fill_mode: str = 'nearest'
    mask_fill_mode: str = 'nearest'
    image_cval: int = 0
    binary_cval: int = 0
    mask_cval: int = 0

    def to_params(self, interp: int, fill_mode: str, cval: float):
        return {
            'rotation_range': self.rotation_range,
            'width_shift_range': self.width_shift_range,
            'height_shift_range': self.height_shift_range,
            'shear_range': self.shear_range,
            'zoom_range': self.zoom_range,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'brightness_range': self.brightness_range,
            'interpolation_order': interp,
            'fill_mode': fill_mode,
            'cval': cval,
        }

    def to_image_params(self):
        return self.to_params(3, self.image_fill_mode, self.image_cval)

    def to_binary_params(self):
        params = self.to_params(0, self.binary_fill_mode, self.binary_cval)
        del params['brightness_range']
        return params

    def to_mask_params(self):
        params = self.to_params(0, self.mask_fill_mode, self.mask_cval)
        del params['brightness_range']
        return params


class TrainSettings(NamedTuple):
    n_epoch: int
    n_classes: int
    l_rate: float
    train_data: Dataset
    validation_data: Dataset
    display: int
    output_dir: str
    threads: int

    data_augmentation: bool = False
    data_augmentation_settings: AugmentationSettings = AugmentationSettings()

    early_stopping_max_performance_drops: int = 10
    early_stopping_restore_best_weights: bool = True
    early_stopping_min_delta: float = 0.0

    reduce_lr_on_plateau: bool = True
    reduce_lr_plateau_factor: float = 0.5
    reduce_lr_min_lr: float = 0.000001

    model_name: str = 'model'
    model_suffix: str = '.h5'
    save_best_model_only: bool = True
    save_weights_only: bool = False

    architecture: Architecture = Architecture.FCN_SKIP
    loss: Loss = Loss.CATEGORICAL_CROSSENTROPY
    monitor: Monitor = Monitor.VAL_LOSS
    optimizer: Optimizers = Optimizers.ADAM

    optimizer_norm_clipping: bool = True
    optimizer_norm_clip_value: float = 1.0
    optimizer_clipping: bool = False
    optimizer_clip_value: float = 1.0
    evaluation_data: Dataset = None

    load: str = None

    continue_training: bool = False
    compute_baseline: bool = False
    foreground_masks: bool = False
    tensorboard: bool = False

    image_dimension: int = 1
    # use the allow_growth tensorflow setting
    # set to true, if you get CUDNN_STATUS_INTERNAL_ERROR, see https://github.com/tensorflow/tensorflow/issues/24496
    gpu_allow_growth: bool = False


class Trainer:
    def __init__(self, settings: TrainSettings):
        self.settings = settings
        tf.keras.backend.clear_session()

        if settings.gpu_allow_growth:
            from ocr4all_pixel_classifier.lib.network import tf_backend_allow_growth
            tf_backend_allow_growth()

        from ocr4all_pixel_classifier.lib.network import Network
        self.train_net = Network("train", settings.n_classes, settings.architecture,
                                 l_rate=settings.l_rate,
                                 foreground_masks=settings.foreground_masks, model=settings.load,
                                 continue_training=settings.continue_training,
                                 input_image_dimension=settings.image_dimension,
                                 optimizer=settings.optimizer,
                                 optimizer_norm_clipping=settings.optimizer_norm_clipping,
                                 optimizer_norm_clip_value=settings.optimizer_norm_clip_value,
                                 optimizer_clipping=settings.optimizer_clipping,
                                 optimizer_clip_value=settings.optimizer_clip_value,
                                 loss_func=settings.loss
                                 )

        if len(settings.train_data) == 0 and settings.n_epoch > 0:
            raise Exception("No training files specified. Maybe set n_iter=0")

        if settings.compute_baseline:
            def compute_label_percentage(label):
                return np.sum([np.sum(d.mask == label) for d in settings.train_data.data]) \
                       / np.sum([d.mask.shape[0] * d.mask.shape[1] for d in settings.train_data.data])

            logging.info("Computing label percentage for {} files.".format(len(settings.train_data.data)))
            label_percentage = [compute_label_percentage(l) for l in range(settings.n_classes)]
            logging.info("Label percentage: {}".format(list(zip(range(settings.n_classes), label_percentage))))
            logging.info("Baseline: {}".format(max(label_percentage)))

    def train(self, callback: Optional[TrainProgressCallback] = None) -> None:
        if callback:
            callback.init(self.settings.n_epoch * len(self.settings.train_data.data),
                          self.settings.early_stopping_max_performance_drops)

        self.train_net.train_dataset(setting=self.settings, callback=callback)

    def eval(self) -> None:
        if self.settings.evaluation_data is None:
            logger.info('Evaluation Dataset in Trainsetting not set! ')
            return
        if len(self.settings.evaluation_data) > 0:
            self.train_net.evaluate_dataset(self.settings.evaluation_data)
        else:
            logger.info('Empty Dataset. Skipping Evaluation')


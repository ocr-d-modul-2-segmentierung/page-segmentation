import tensorflow as tf
import numpy as np
from typing import Optional
from pagesegmentation.lib.callback import TrainProgressCallback, TrainProgressCallbackWrapper
from pagesegmentation.lib.trainer import TrainSettings, AugmentationSettings
from pagesegmentation.lib.util import image_to_batch, gray_to_rgb
from .dataset import Dataset, SingleData
import os
import logging
from pagesegmentation.lib.model import Optimizers, Architecture

logger = logging.getLogger(__name__)


class Network:
    def __init__(self,
                 type: str,

                 n_classes: int = -1,
                 model_constructor: Architecture = Architecture.FCN_SKIP,
                 l_rate: float = 1e-4,
                 has_binary: bool = False,
                 foreground_masks: bool = False,
                 model: str = None,
                 continue_training: bool = False,
                 input_image_dimension: int = 1,
                 optimizer: Optimizers = Optimizers.ADAM,
                 optimizer_norm_clipping: bool = True,
                 optimizer_norm_clip_value: float = 1.0,
                 optimizer_clipping=False,
                 optimizer_clip_value=1,
                 loss_func=None
                 ):
        """
        :param model_constructor: function that takes the input layer and number of classes and creates the model
        :param n_classes: number of classes
        :param l_rate: learning rate
        :param has_binary:
        :param fixed_size: resize images to given dimensions
        :param data_augmentation: preprocessing to apply to data
        :param foreground_masks: keep only mask parts that are foreground in binary image (training only)
        :param model: continue Training
        :param
        """
        self.architecture = model_constructor.value
        self._data: Dataset = Dataset([], {})
        self.type = type
        self.has_binary = has_binary
        self.foreground_masks = foreground_masks

        self.binary = tf.keras.layers.Input((None, None, 1))
        self.n_classes = n_classes
        preprocess, rgb = Architecture(self.architecture).preprocess()
        if rgb:
            self.input = tf.keras.layers.Input((None, None, 3))
        else:
            self.input = tf.keras.layers.Input((None, None, input_image_dimension))

        model = model if not model or '.' in model else model + '.h5'
        if model and not os.path.exists(model) and model.endswith('.h5'):
            from subprocess import check_call
            import sys
            from pathlib import Path
            logger.info("Upgrading model to {}".format(model))
            check_call([sys.executable, os.path.join(Path(__file__).parent.parent, 'scripts', 'migrate_model.py'),
                        '--meta_path', model[:-3] + '.meta', '--output_path', model,
                        '--n_classes', str(n_classes), '--l_rate', str(l_rate)
                        ])

        from pagesegmentation.lib.metrics import accuracy, loss, dice_coef, \
            fgpa, fgpl, jacard_coef, dice_coef_loss, jacard_coef_loss, categorical_hinge, dice_and_categorical\
            , categorical_focal_loss
        from pagesegmentation.lib.layers import GraytoRgb

        try:
            self.model = tf.keras.models.load_model(model, custom_objects={'loss': loss, 'accuracy': accuracy,
                                                                           'dice_coef': dice_coef,
                                                                           'jacard_coef': jacard_coef,
                                                                           'dice_coef_loss': dice_coef_loss,
                                                                           'jacard_coef_loss': jacard_coef_loss,
                                                                           'dice_and_categorical': dice_and_categorical,
                                                                           'categorical_hinge': categorical_hinge,
                                                                           'categorical_focal_loss':categorical_focal_loss,
                                                                           'GraytoRgb': GraytoRgb})
        except Exception as e:
            if model and continue_training:
                raise e

            self.model = model_constructor.model()([self.input], n_classes)
            optimizer = optimizer()
            _optimizer = None
            if optimizer_norm_clipping and optimizer_clipping:
                _optimizer = optimizer(lr=l_rate,
                                       clipnorm=optimizer_norm_clip_value,
                                       clipvalue=optimizer_clip_value)
            else:
                if optimizer_norm_clipping:
                    _optimizer = optimizer(lr=l_rate, clipnorm=optimizer_norm_clip_value)
                elif optimizer_clipping:
                    _optimizer = optimizer(lr=l_rate, clipvalue=optimizer_clip_value)
                else:
                    _optimizer = optimizer(lr=l_rate)

            if self.type == "train":
                self.model.compile(optimizer=_optimizer, loss=loss_func(), metrics=[accuracy, jacard_coef, dice_coef])

            if model:
                self.model.load_weights(model)

    def create_dataset_inputs(self, train_data, data_augmentation=True,
                              data_augmentation_settings: AugmentationSettings = AugmentationSettings()):
        preprocess, rgb = Architecture(self.architecture).preprocess()

        while True:
            for data_idx, d in enumerate(train_data):

                b, i, m = d.binary, d.image, d.mask
                if rgb:
                    i = gray_to_rgb(i)

                if b is None:
                    b = np.full(i.shape, 1, dtype=np.uint8)
                    assert (i.dtype == np.uint8)
                if self.foreground_masks:
                    m[b != 1] = 0
                if self.type == 'train' and data_augmentation:
                    from pagesegmentation.lib.data_generator import ImageDataGeneratorCustom
                    image_gen = ImageDataGeneratorCustom(**data_augmentation_settings._asdict(),
                                                         fill_mode='nearest',
                                                         data_format='channels_last',
                                                         interpolation_order=1)

                    binary_gen = ImageDataGeneratorCustom(rotation_range=data_augmentation_settings.rotation_range,
                                                          width_shift_range=data_augmentation_settings.width_shift_range,
                                                          height_shift_range=data_augmentation_settings.height_shift_range,
                                                          shear_range=data_augmentation_settings.shear_range,
                                                          zoom_range=data_augmentation_settings.zoom_range,
                                                          horizontal_flip=data_augmentation_settings.horizontal_flip,
                                                          vertical_flip=data_augmentation_settings.vertical_flip,
                                                          fill_mode='nearest',
                                                          data_format='channels_last',
                                                          interpolation_order=0)

                    mask_gen = ImageDataGeneratorCustom(rotation_range=data_augmentation_settings.rotation_range,
                                                        width_shift_range=data_augmentation_settings.width_shift_range,
                                                        height_shift_range=data_augmentation_settings.height_shift_range,
                                                        shear_range=data_augmentation_settings.shear_range,
                                                        zoom_range=data_augmentation_settings.zoom_range,
                                                        horizontal_flip=data_augmentation_settings.horizontal_flip,
                                                        vertical_flip=data_augmentation_settings.vertical_flip,
                                                        fill_mode='nearest',
                                                        data_format='channels_last',
                                                        interpolation_order=0,
                                                        )
                    seed = np.random.randint(0, 9999999)
                    i_x = image_gen.flow(image_to_batch(i), seed=seed, batch_size=1)
                    b_x = binary_gen.flow(image_to_batch(b), seed=seed, batch_size=1)
                    m_x = mask_gen.flow(image_to_batch(m), seed=seed, batch_size=1)

                    i_n = next(i_x)
                    b_n = next(b_x)
                    m_n = next(m_x)

                    yield [preprocess(i_n), b_n], m_n
                else:
                    yield [image_to_batch(preprocess(i)),
                           image_to_batch(b)], \
                          image_to_batch(m)

    def train_dataset(self, setting: TrainSettings = None,
                      callback: Optional[TrainProgressCallback] = None):
        logger.info(self.model.summary)

        import os
        callbacks = []
        train_gen = self.create_dataset_inputs(setting.train_data, setting.data_augmentation)
        test_gen = self.create_dataset_inputs(setting.validation_data, data_augmentation=False)

        os.makedirs(setting.output_dir, exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(setting.output_dir, setting.model_name +
                                                                     setting.model_suffix),
                                                        monitor=setting.monitor.value,
                                                        verbose=1,
                                                        save_best_only=setting.save_best_model_only,
                                                        save_weights_only=setting.save_weights_only)
        callbacks.append(checkpoint)

        if setting.early_stopping_max_performance_drops != 0:
            early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=setting.monitor.value,
                                                             patience=setting.early_stopping_max_performance_drops,
                                                             verbose=1, mode='auto',
                                                             restore_best_weights=
                                                             setting.early_stopping_restore_best_weights,
                                                             min_delta=setting.early_stopping_min_delta)
            callbacks.append(early_stop_cb)
        else:
            early_stop_cb = None

        if setting.tensorboard:
            from pagesegmentation.lib.callback import ModelDiagnoser
            import pathlib
            import datetime
            now = datetime.datetime.today()
            output = os.path.join(
                setting.output_dir, 'logs', now.strftime('%Y-%m-%d_%H-%M-%S'))
            pathlib.Path(output).mkdir(parents=True,
                                       exist_ok=True)
            callback_gen = self.create_dataset_inputs(setting.validation_data,
                                                      data_augmentation=False)

            diagnose_cb = ModelDiagnoser(callback_gen,  # data_generator
                                         1,  # batch_size
                                         len(setting.validation_data),  # num_samples
                                         output,  # output_dir
                                         setting.validation_data.color_map)  # color_map

            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=output + '/logs',
                                                         histogram_freq=1,
                                                         batch_size=1,
                                                         write_graph=True,
                                                         write_images=False)
            callbacks.append(diagnose_cb)
            callbacks.append(tensorboard)

        if setting.reduce_lr_on_plateau:
            redurce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=setting.monitor.value,
                factor=setting.reduce_lr_plateau_factor,
                patience=setting.early_stopping_max_performance_drops / 2,
                min_lr=setting.reduce_lr_min_lr,
                verbose=1)
            callbacks.append(redurce_lr_plateau)

        if callback:
            callbacks.append(TrainProgressCallbackWrapper(
                len(setting.train_data),
                callback,
                early_stop_cb,
            ))

        fg = self.model.fit(train_gen,
                            epochs=setting.n_epoch,
                            steps_per_epoch=len(setting.train_data),
                            use_multiprocessing=False,
                            workers=1,
                            validation_steps=len(setting.validation_data),
                            validation_data=test_gen,
                            callbacks=callbacks)

        return fg

    def evaluate_dataset(self, eval_data):
        eval_gen = self.create_dataset_inputs(eval_data, data_augmentation=False)
        self.model.evaluate(eval_gen, batch_size=1, steps=len(eval_data))

    def predict_single_data(self, data: SingleData):
        from scipy.special import softmax
        image = data.image
        preprocess, rgb = Architecture(self.architecture).preprocess()
        if rgb:
            image = gray_to_rgb(image)
        logit = self.model.predict([image_to_batch(preprocess(image)),
                                   image_to_batch(data.binary)])[0, :, :, :]
        prob = softmax(logit, -1)
        pred = np.argmax(logit, -1)
        return logit, prob, pred


def tf_backend_allow_growth():
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

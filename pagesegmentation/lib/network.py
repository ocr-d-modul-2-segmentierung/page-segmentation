import tensorflow as tf
import numpy as np
from typing import Optional
from pagesegmentation.lib.callback import TrainProgressCallback, TrainProgressCallbackWrapper

from .dataset import Dataset, SingleData
import os
import logging

logger = logging.getLogger(__name__)


class Network:
    def __init__(self,
                 type: str,
                 model_constructor,
                 n_classes: int,
                 l_rate: float = 1e-4,
                 has_binary: bool = False,
                 foreground_masks: bool = False,
                 model: str = None,
                 continue_training: bool = False,
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
        self._data: Dataset = Dataset([], {})
        self.type = type
        self.has_binary = has_binary
        self.foreground_masks = foreground_masks
        self.input = tf.keras.layers.Input((None, None, 1))
        self.binary = tf.keras.layers.Input((None, None, 1))
        self.n_classes = n_classes
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
            fgpa, fgpl, jacard_coef, dice_coef_loss, jacard_coef_loss
        if model and continue_training:
            self.model = tf.keras.models.load_model(model, custom_objects={'loss': loss, 'accuracy': accuracy,
                                                                           'fgpa': fgpa, 'fgpl': fgpl,
                                                                           'dice_coef': dice_coef,
                                                                           'jacard_coef': jacard_coef,
                                                                           'dice_coef_loss': dice_coef_loss,
                                                                           'jacard_coef_loss': jacard_coef_loss})
        else:
            self.model = model_constructor([self.input, self.binary], n_classes)
            optimizer = tf.keras.optimizers.Adam(lr=l_rate)
            self.model.compile(optimizer=optimizer, loss=loss(n_classes), metrics=[accuracy, fgpa(self.binary),
                                                                        jacard_coef, dice_coef])
            if model:
                self.model.load_weights(model)

    def create_dataset_inputs(self, train_data, data_augmentation=True):
        def gray_to_rgb(img):
            return np.repeat(img, 3, 2)

        def preprocessing(array):
            frac, integ = np.modf(array)
            mask = (frac > 0) & (frac < 0.5)
            array[mask] = 0
            return np.round(array)

        while True:
            for data_idx, d in enumerate(train_data):

                b, i, m = d.binary, d.image, d.mask
                if b is None:
                    b = np.full(i.shape, 1, dtype=np.uint8)
                    assert (i.dtype == np.uint8)
                if self.foreground_masks:
                    m[b != 1] = 0
                if self.type == 'train' and data_augmentation:
                    from pagesegmentation.lib.data_generator import ImageDataGeneratorCustom
                    image_gen = ImageDataGeneratorCustom(rotation_range=2.5,
                                                         width_shift_range=0.025,
                                                         height_shift_range=0.025,
                                                         shear_range=0.00,
                                                         zoom_range=[0.95, 1.05],
                                                         horizontal_flip=False,
                                                         vertical_flip=False,
                                                         fill_mode='nearest',
                                                         data_format='channels_last',
                                                         brightness_range=[0.95, 1.05],
                                                         interpolation_order=1)

                    binary_gen = ImageDataGeneratorCustom(rotation_range=2.5,
                                                          width_shift_range=0.025,
                                                          height_shift_range=0.025,
                                                          shear_range=0.00,
                                                          zoom_range=[0.95, 1.05],
                                                          horizontal_flip=False,
                                                          vertical_flip=False,
                                                          fill_mode='nearest',
                                                          data_format='channels_last',
                                                          interpolation_order=0)

                    mask_gen = ImageDataGeneratorCustom(rotation_range=2.5,
                                                        width_shift_range=0.025,
                                                        height_shift_range=0.025,
                                                        shear_range=0.00,
                                                        zoom_range=[0.95, 1.05],
                                                        horizontal_flip=False,
                                                        vertical_flip=False,
                                                        fill_mode='nearest',
                                                        data_format='channels_last',
                                                        interpolation_order=0,
                                                        )
                    seed = np.random.randint(0, 9999999)
                    i_x = image_gen.flow(np.expand_dims(np.expand_dims(i, axis=0), axis=-1), seed=seed, batch_size=1)
                    b_x = binary_gen.flow(np.expand_dims(np.expand_dims(b, axis=0), axis=-1), seed=seed, batch_size=1)
                    m_x = mask_gen.flow(np.expand_dims(np.expand_dims(m, axis=0), axis=-1), seed=seed, batch_size=1)

                    i_n = next(i_x)
                    b_n = next(b_x)
                    m_n = next(m_x)

                    yield [i_n / 255.0, b_n], m_n
                else:
                    yield [np.expand_dims(np.expand_dims(i / 255.0, axis=0), axis=-1),
                           np.expand_dims(np.expand_dims(b, axis=0), axis=-1)], \
                          np.expand_dims(np.expand_dims(m, axis=0), axis=-1)

    def train_dataset(self, train_data: Dataset, test_data: Dataset,
                      output_dir: str, best_model_name: str,
                      epochs: int = 100,
                      early_stopping: bool = True, early_stopping_interval: int = 5, tensorboardlogs: bool = True,
                      augmentation: bool = False, reduce_lr_on_plateu=False,
                      callback: Optional[TrainProgressCallback] = None):
        import os
        callbacks = []
        train_gen = self.create_dataset_inputs(train_data, augmentation)
        test_gen = self.create_dataset_inputs(test_data, data_augmentation=False)
        logger.info(self.model.summary)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, best_model_name + '.h5'),
                                                        monitor='val_accuracy',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=True)
        callbacks.append(checkpoint)

        if early_stopping:
            early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                             patience=early_stopping_interval,
                                                             min_delta=0.0005,
                                                             verbose=1, mode='auto',
                                                             restore_best_weights=True)
            callbacks.append(early_stop_cb)
        else:
            early_stop_cb = None

        if tensorboardlogs:
            from pagesegmentation.lib.callback import ModelDiagnoser
            import pathlib
            import datetime
            now = datetime.datetime.today()
            output = os.path.join(
                output_dir, 'logs', now.strftime('%Y-%m-%d_%H-%M-%S'))
            pathlib.Path(output).mkdir(parents=True,
                                       exist_ok=True)
            callback_gen = self.create_dataset_inputs(test_data,
                                                      data_augmentation=False)

            diagnose_cb = ModelDiagnoser(callback_gen,  # data_generator
                                         1,  # batch_size
                                         len(test_data),  # num_samples
                                         output,  # output_dir
                                         test_data.color_map)  # color_map

            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=output + '/logs',
                                                         histogram_freq=1,
                                                         batch_size=1,
                                                         write_graph=True,
                                                         write_images=False)
            callbacks.append(diagnose_cb)
            callbacks.append(tensorboard)

        if reduce_lr_on_plateu:
            redurce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=early_stopping_interval / 2,
                min_lr=0.000001,
                verbose=1)
            callbacks.append(redurce_lr_plateau)

        if callback:
            callbacks.append(TrainProgressCallbackWrapper(
                len(train_data),
                callback,
                early_stop_cb,
            ))

        fg = self.model.fit(train_gen,
                            epochs=epochs,
                            steps_per_epoch=len(train_data),
                            use_multiprocessing=False,
                            workers=1,
                            validation_steps=len(test_data),
                            validation_data=test_gen,
                            callbacks=callbacks)

        return fg

    def evaluate_dataset(self, eval_data):
        eval_gen = self.create_dataset_inputs(eval_data, data_augmentation=False)
        self.model.evaluate(eval_gen, batch_size=1, steps=len(eval_data))

    def predict_single_data(self, data: SingleData):
        from scipy.special import softmax
        logit = self.model.predict([np.expand_dims(np.expand_dims(data.image / 255, axis=0), axis=-1),
                                   np.expand_dims(np.expand_dims(data.binary, axis=0), axis=-1)])[0, :, :, :]
        prob = softmax(logit, -1)
        pred = np.argmax(logit, -1)
        return logit, prob, pred


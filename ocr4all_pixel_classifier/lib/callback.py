import numpy as np
import tensorflow as tf
import os
from ocr4all_pixel_classifier.lib.util import image_to_batch, gray_to_rgb
from ocr4all_pixel_classifier.lib.data_generator import DataGenerator
import matplotlib.pyplot as plt

class TrainProgressCallback(tf.keras.callbacks.Callback):
    def init(self, total_iters, early_stopping_iters):
        pass

    def update_loss(self, batch: int, loss: float, acc: float):
        pass

    def next_best(self, epoch, acc, n_best):
        pass


class TrainProgressCallbackWrapper(tf.keras.callbacks.Callback):

    def __init__(self,
                 n_iters_per_epoch: int,
                 train_callback: TrainProgressCallback,
                 early_stopping_callback=None):
        super().__init__()
        self.train_callback = train_callback
        self.early_stopping_callback = early_stopping_callback
        self.n_iters_per_epoch = n_iters_per_epoch
        self.epoch = 0
        self.iter = 0

    def on_batch_end(self, batch, logs=None):
        self.iter = batch + self.epoch * self.n_iters_per_epoch
        self.train_callback.update_loss(self.iter,
                                        logs.get('loss'),
                                        logs.get('accuracy'),
                                        )

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1
        if self.early_stopping_callback:
            self.train_callback.next_best(self.iter, self.early_stopping_callback.best, self.early_stopping_callback.wait)


class TensorboardWriter:

    def __init__(self, outdir, max_outputs=10):
        assert (os.path.isdir(outdir))
        self.outdir = outdir
        self.writer = tf.summary.create_file_writer(self.outdir, flush_millis=10000)
        self.counter: int = 0
        self.max_outputs=max_outputs

    def save_image(self, tag, image, global_step=None):
        with self.writer.as_default():
            tf.summary.image(
                tag,
                image,
                step = self.counter,
                max_outputs=self.max_outputs,
            )
        self.counter += 1

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


class ModelDiagnoser(tf.keras.callbacks.Callback):

    def __init__(self, data_generator:DataGenerator, output_dir):
        super().__init__()
        self.data_generator = data_generator
        self.batch_size = data_generator.batch_size
        self.num_samples = len(data_generator)
        self.tensorboard_writer = TensorboardWriter(output_dir)
        self.color_map = self.data_generator.data_set.color_map

    def on_epoch_end(self, epoch, logs=None):
        from ocr4all_pixel_classifier.lib.dataset import label_to_colors
        sample_index = 0
        for i in range(self.num_samples):
            generator_output = self.data_generator[i]
            x, y = generator_output
            logit = self.model.predict_on_batch(x)[0, :, :, :]
            pred = np.argmax(logit, -1)
            color_mask = label_to_colors(pred, colormap=self.color_map)
            inv_binary = np.stack([x.get('input_2')[0, :, :, 0]] * 3, axis=-1)
            inverted_overlay_mask = color_mask.copy()
            inverted_overlay_mask[inv_binary == 0] = 0
            self.tensorboard_writer.save_image("{}/Input"
                                                   .format(sample_index, epoch), tf.convert_to_tensor(image_to_batch(x.get('input_1')[0, :, :, :])))
            self.tensorboard_writer.save_image("{}/GT"
                                                   .format(sample_index, epoch), tf.convert_to_tensor(image_to_batch(label_to_colors(y.get('logits')[0, :, :, 0],
                                                                                                 self.color_map))))
            self.tensorboard_writer.save_image("{}/Prediction"
                                                   .format(sample_index, epoch), tf.convert_to_tensor(image_to_batch(color_mask)))
            self.tensorboard_writer.save_image("{}/Overlay"
                                                   .format(sample_index, epoch), tf.convert_to_tensor(image_to_batch(inverted_overlay_mask)))
            n0, n1, n2 = np.shape(logit)
            for i in range(n2):
                image = logit[:,:,i]
                image -= np.min(image)  # ensure the minimal value is 0.0
                image /= np.max(image)  # maximum value in image is now 1.0
                cm = plt.get_cmap('jet')
                cmap = cm(image)
                self.tensorboard_writer.save_image("{}/Heatmap: Class: {}"
                                                   .format(sample_index, i),
                                                   tf.convert_to_tensor(image_to_batch(cmap)))
            sample_index += 1

    def on_train_end(self, logs=None):
        self.tensorboard_writer.close()

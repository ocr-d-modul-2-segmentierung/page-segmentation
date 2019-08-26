import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os


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


def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    else:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorboardWriter:

    def __init__(self, outdir):
        assert (os.path.isdir(outdir))
        self.outdir = outdir
        self.writer = tf.summary.FileWriter(self.outdir,
                                            flush_secs=10)

    def save_image(self, tag, image, global_step=None):
        image_tensor = make_image_tensor(image)
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, image=image_tensor)]),
                                global_step)

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


class ModelDiagnoser(tf.keras.callbacks.Callback):

    def __init__(self, data_generator, batch_size, num_samples, output_dir, color_map):
        super().__init__()
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.tensorboard_writer = TensorboardWriter(output_dir)
        self.color_map = color_map

    def on_epoch_end(self, epoch, logs=None):
        from pagesegmentation.lib.dataset import label_to_colors
        total_steps = int(np.ceil(np.divide(self.num_samples, self.batch_size)))
        sample_index = 0
        while sample_index < total_steps:
            generator_output = next(self.data_generator)

            x, y = generator_output
            logit = self.model.predict(x)[0, :, :, :]
            pred = np.argmax(logit, -1)
            color_mask = label_to_colors(pred, colormap=self.color_map)
            inv_binary = np.stack([x[1][0, :, :, 0]] * 3, axis=-1)
            inverted_overlay_mask = color_mask.copy()
            inverted_overlay_mask[inv_binary == 0] = 0
            self.tensorboard_writer.save_image("{}/Input"
                                                   .format(sample_index, epoch), x[0][0, :, :, 0] * 255)
            self.tensorboard_writer.save_image("{}/GT"
                                                   .format(sample_index, epoch), label_to_colors(y[0, :, :, 0],
                                                                                                 self.color_map))
            self.tensorboard_writer.save_image("{}/Prediction"
                                                   .format(sample_index, epoch), color_mask)
            self.tensorboard_writer.save_image("{}/Overlay"
                                                   .format(sample_index, epoch), inverted_overlay_mask)

            sample_index += 1

    def on_train_end(self, logs=None):
        self.tensorboard_writer.close()
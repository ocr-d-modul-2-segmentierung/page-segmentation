import tensorflow as tf
import numpy as np
import sys
from typing import Generator, Tuple

from .data_augmenter import DataAugmenterBase
from .dataset import Dataset, SingleData


class Network:
    def __init__(self,
                 type: str,
                 graph: tf.Graph,
                 session: tf.Session,
                 model_constructor,
                 n_classes: int,
                 l_rate: float,
                 reuse: bool = False, has_binary: bool = False, fixed_size: Tuple[int, int] = None,
                 data_augmentation: DataAugmenterBase = None,
                 meta: str = '', foreground_masks: bool = False):
        """
        :param type: network type, either "train", "test", "deploy" or "meta" (for loading from file)
        :param graph: tensorflow graph
        :param session: tensorflow session
        :param model_constructor: function that takes the input layer and number of classes and creates the model
        :param n_classes: number of classes
        :param l_rate: learning rate
        :param reuse: reuse tensorflow variable scope
        :param has_binary:
        :param fixed_size: resize images to given dimensions
        :param data_augmentation: preprocessing to apply to data
        :param meta: name of model to load (if type = meta)
        :param foreground_masks: keep only mask parts that are foreground in binary image (training only)
        """
        meta = meta[:-5] if meta.endswith(".meta") else meta
        self.type = type
        self._data: Dataset = Dataset([])
        self.graph = graph
        self.session = session
        self.has_binary = has_binary
        self.data_augmentation = data_augmentation
        with self.graph.as_default():
            if type == "meta":
                self._init_from_meta(graph, meta)
            else:
                with tf.variable_scope("network", reuse=reuse):
                    if type == "train" or type == "test":
                        self.raw_binary_inputs, self.raw_inputs, self.raw_masks, self.data_idx, self.data_initializer = self.create_dataset_inputs()
                    elif type == "deploy":
                        self.raw_binary_inputs, self.raw_inputs, self.raw_masks = self.create_placeholders()
                        self.data_idx = None
                        self.data_initializer = None
                    else:
                        raise Exception("Undefined network")

                    if fixed_size:
                        h, w = fixed_size
                        self.inputs = tf.image.resize_image_with_crop_or_pad(tf.expand_dims(self.raw_inputs, [-1]), h,
                                                                             w)
                        self.binary_inputs = tf.squeeze(
                            tf.image.resize_image_with_crop_or_pad(tf.expand_dims(self.raw_binary_inputs, [-1]), h, w),
                            axis=-1)
                        self.masks = tf.squeeze(
                            tf.image.resize_image_with_crop_or_pad(tf.expand_dims(self.raw_masks, [-1]), h, w), axis=-1)
                    else:
                        self.inputs = tf.expand_dims(self.raw_inputs, [-1])
                        self.binary_inputs = self.raw_binary_inputs
                        self.masks = self.raw_masks

                    self.inputs = tf.cast(self.inputs, tf.float32) / 255.0

                    self.logits = model_constructor(self.inputs, n_classes)
                    self.logits = tf.identity(self.logits, name="logits")
                    self.probs = tf.nn.softmax(self.logits, -1, name="probabilities")
                    self.prediction = tf.argmax(self.logits, axis=-1, name="prediction")

                    if type == "train" or type == "test":
                        self.single_pa, self.pixel_accuracy, self.single_fgpa, self.foreground_pixel_accuracy = \
                            self.create_errors(self.prediction, self.masks, self.binary_inputs)

                        self.loss, self.train_op = self.create_solver(self.logits, self.binary_inputs, self.masks,
                                                                      l_rate, foreground_masks)
                    else:
                        self.pixel_accuracy = None
                        self.foreground_pixel_accuracy = None

                    self.session.run(tf.global_variables_initializer())

        tf.reset_default_graph()

    def _init_from_meta(self, graph, meta):
        saver = tf.train.import_meta_graph(meta + '.meta')
        saver.restore(self.session, meta)
        self.raw_binary_inputs = self.raw_masks = self.data_idx = self.data_initializer = None
        self.raw_inputs = graph.get_tensor_by_name("network/inputs:0")
        self.probs = graph.get_tensor_by_name("network/probabilities:0")
        self.prediction = graph.get_tensor_by_name("network/prediction:0")

    def set_data(self, data: Dataset):
        self._data = data

    @staticmethod
    def create_solver(logits, binary_inputs, masks, l_rate, foreground_masks: bool, solver="Adam"):
        if foreground_masks:
            masks = masks * binary_inputs

        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(masks, tf.int32),
            logits=logits,
            # weights=binary_inputs * 0 + 1,
        ))

        if solver == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate=l_rate, momentum=0.9)

        gvs = optimizer.compute_gradients(loss)
        grads = [grad for grad, _ in gvs]
        global_norm = True
        if global_norm:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1)
        else:
            grads = [tf.clip_by_value(grad, -0.1, 0.1) for grad in grads]

        gvs = zip(grads, [var for _, var in gvs])
        train_op = optimizer.apply_gradients(gvs, name="train_op")

        return loss, train_op

    @staticmethod
    def create_errors(prediction, masks, binary_inputs, batch_size=1):
        equals = tf.equal(tf.cast(prediction, tf.uint8), masks)
        single_accuracy = tf.reduce_mean(tf.cast(equals, tf.float32), axis=[1, 2])
        accuracy = tf.reduce_mean(single_accuracy)

        def count_fg(img):
            return tf.reduce_sum(tf.cast(tf.reshape(img, (batch_size, -1)), tf.int32), axis=-1)

        fg_equals = tf.multiply(tf.cast(equals, tf.uint8), binary_inputs)
        fgpa_correct = count_fg(fg_equals)
        fgpa_total = count_fg(binary_inputs)

        single_fgpa = tf.divide(tf.cast(fgpa_correct, tf.float32), tf.cast(fgpa_total, tf.float32))
        fgpa = tf.reduce_mean(single_fgpa)

        return single_accuracy, accuracy, single_fgpa, fgpa

    def create_dataset_inputs(self, batch_size=1, buffer_size=50):
        with tf.variable_scope("", reuse=False):
            data_augmenter = self.data_augmentation

            def gen():
                for data_idx, d in enumerate(self._data):
                    b, i, m = d.binary, d.image, d.mask
                    if b is None:
                        b = np.full(i.shape, 1, dtype=np.uint8)

                    assert(b.dtype == np.uint8)
                    assert(i.dtype == np.uint8)
                    assert(m.dtype == np.uint8)
                    print(b.shape, i.shape, i.min(), i.max(), m.min(), m.max())
                    yield b, i, m, data_idx

            def data_augmentation(b, i, m, data_idx):
                return data_augmenter.apply(b, i, m) + (data_idx,)

            def set_data_shapes(b, i, m, data_idx):
                b.set_shape([None, None])
                i.set_shape([None, None])
                m.set_shape([None, None])
                return b, i, m, data_idx

            datatype = (tf.uint8, tf.uint8, tf.uint8, tf.int32)
            dataset = tf.data.Dataset.from_generator(gen, datatype, ([None, None], [None, None], [None, None], None))
            if self.type == "train":
                if data_augmenter:
                    # map must not one thread less then the inter threads
                    dataset = dataset.map(
                        lambda b, i, m, data_idx: tuple(tf.py_func(data_augmentation, (b, i, m, data_idx), datatype)),
                        num_parallel_calls=max(1, self.session._config.inter_op_parallelism_threads - 1))
                    dataset = dataset.map(set_data_shapes)

                dataset = dataset.repeat().shuffle(buffer_size, seed=10)
            else:
                pass

            dataset = dataset.batch(batch_size)
            # data augmentation
            # dataset = dataset.map(convert_to_sparse)

            data_initializer = dataset.prefetch(20).make_initializable_iterator()
            inputs = data_initializer.get_next()
            return inputs[0], inputs[1], inputs[2], inputs[3], data_initializer

    def create_placeholders(self, b=1):
        if self.has_binary:
            raw_binary_inputs = tf.placeholder(tf.uint8, (b, None, None), "binary_inputs")
        else:
            raw_binary_inputs = None

        raw_inputs = tf.placeholder(tf.uint8, (b, None, None), "inputs")
        raw_masks = tf.placeholder(tf.uint8, (b, None, None), "masks")

        return raw_binary_inputs, raw_inputs, raw_masks

    def uninitialized_variables(self):
        with self.graph.as_default():
            global_vars = tf.global_variables()
            is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            return not_initialized_vars

    def prepare(self, uninitialized_variables_only=True):
        with self.graph.as_default():
            if uninitialized_variables_only:
                self.session.run(tf.variables_initializer(self.uninitialized_variables()))
            else:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                self.session.run(init_op)

            self.session.run([self.data_initializer.initializer])

    def load_weights(self, filepath, restore_only_trainable=True):
        with self.graph.as_default():
            # reload trainable variables only (e. g. omitting solver specific variables)
            if restore_only_trainable:
                saver = tf.train.Saver(tf.trainable_variables())
            else:
                saver = tf.train.Saver()

            # Restore variables from disk.
            # This will possible load a weight matrix with wrong shape, thus a codec resize is necessary
            saver.restore(self.session, filepath)

    def save_checkpoint(self, output_file, checkpoint=None):
        with self.graph.as_default():
            saver = tf.train.Saver()
            if checkpoint is None:
                saver.save(self.session, output_file)
            else:
                saver.save(self.session, output_file, global_step=checkpoint)

    def train_dataset(self):
        l, _, a, fg = self.session.run((self.loss, self.train_op, self.pixel_accuracy, self.foreground_pixel_accuracy))

        if not np.isfinite(l):
            print("WARNING: Infinite loss. Skipping batch.", file=sys.stderr)

        return l, a, fg

    def test_dataset(self) -> Generator[Tuple[float, float, float, SingleData], None, None]:
        with self.graph.as_default():
            if self.data_initializer:
                self.session.run([self.data_initializer.initializer])

        try:
            while True:
                pred, acc, fgacc, idxs = self.session.run(
                    (self.prediction, self.single_pa, self.single_fgpa, self.data_idx))
                for l, a, fg, idx in zip(pred, acc, fgacc, idxs):
                    yield l, a, fg, self._data.data[idx]

        except tf.errors.OutOfRangeError:
            pass

    def predict_single_data(self, data: SingleData):
        pred, prob = self.session.run((self.prediction, self.probs), {
            self.raw_inputs: [data.image],
        })
        return pred[0], prob[0]

    def iters_per_epoch(self):
        return len(self._data)  # / batch_size

    def n_data(self):
        return len(self._data)

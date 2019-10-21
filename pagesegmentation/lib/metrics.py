import tensorflow as tf
import logging
import enum

logger = logging.getLogger(__name__)


def loss(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))


def accuracy(y_true, y_pred):
    n_classes = tf.keras.backend.shape(y_pred)[3]
    y_true = tf.keras.backend.reshape(y_true, (-1,))
    y_pred = tf.keras.backend.reshape(y_pred, (-1, n_classes))
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.cast(y_true, 'int64'),
                                                        tf.keras.backend.argmax(y_pred, axis=-1)))


def fgpl(k):
    binary = k[0, :, :, 0]

    def fgpa_loss(y_true, y_pred):
        n_classes = tf.keras.backend.shape(y_pred)[3]
        bin = tf.keras.backend.reshape(binary, (-1,))
        bin_classes = tf.keras.backend.reshape(tf.keras.backend.concatenate(
            [bin for i in range(n_classes)], axis=-1), (-1, n_classes))
        y_true = tf.keras.backend.reshape(y_true, (-1,)) * bin
        y_pred = tf.keras.backend.reshape(y_pred, (-1, n_classes)) * bin_classes
        return tf.keras.backend.mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred,
                                                                                     from_logits=True))

    return fgpa_loss


def fgpa(binary_inputs):
    binary = tf.keras.backend.reshape(binary_inputs, (-1,))

    def fgpa_accuracy(y_true, y_pred):
        n_classes = tf.keras.backend.shape(y_pred)[3]
        y_true = tf.keras.backend.reshape(y_true, (-1,))
        y_pred = tf.keras.backend.reshape(y_pred, (-1, n_classes))
        equals = tf.keras.backend.equal(tf.keras.backend.cast(y_true, 'int64'),
                                        tf.keras.backend.argmax(y_pred, axis=-1))

        def count_fg(img):
            return tf.reduce_sum(input_tensor=img, axis=-1)

        fg_equals = tf.multiply(tf.keras.backend.cast(equals, 'int64'), tf.keras.backend.cast(binary, 'int64'))
        fgpa_correct = count_fg(fg_equals)
        fgpa_total = count_fg(binary)
        single_fgpa = tf.divide(tf.cast(fgpa_correct, tf.float32), tf.cast(fgpa_total, tf.float32))
        fgpa = tf.reduce_mean(input_tensor=single_fgpa)

        return fgpa

    return fgpa_accuracy


def jacard_coef(y_true, y_pred):
    n_classes = tf.keras.backend.shape(y_pred)[3]
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = tf.keras.backend.squeeze(y_true, axis=-1)
    y_true = tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int64'), n_classes)

    intersection = tf.reduce_sum(input_tensor=tf.keras.backend.abs(y_true * y_pred), axis=(1, 2))
    sum_ = tf.reduce_sum(input_tensor=tf.keras.backend.abs(y_true + y_pred), axis=(1, 2))
    jac = (intersection + 100) / (sum_ - intersection + 100)
    return tf.reduce_mean(input_tensor=jac, axis=0)


def jacard_coef_loss(y_true, y_pred):
    return -tf.math.log(jacard_coef(y_true, y_pred))


def dice_coef(y_true, y_pred):
    n_classes = tf.keras.backend.shape(y_pred)[3]
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = tf.keras.backend.squeeze(y_true, axis=-1)
    y_true = tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int64'), n_classes)

    intersection = tf.reduce_sum(input_tensor=tf.keras.backend.abs(y_true * y_pred), axis=(1, 2))
    sum_ = tf.reduce_sum(input_tensor=tf.keras.backend.abs(y_true + y_pred), axis=(1, 2))
    dice = (2.0 * intersection + 100) / (sum_ + 100)
    return tf.reduce_mean(input_tensor=dice, axis=0)


def dice_coef_loss(y_true, y_pred):
    return -tf.math.log(dice_coef(y_true, y_pred))


def categorical_hinge(y_true, y_pred):
    n_classes = tf.keras.backend.shape(y_pred)[3]
    y_true = tf.keras.backend.squeeze(y_true, axis=-1)
    y_true = tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int64'), n_classes)
    pos = tf.keras.backend.sum(y_true * y_pred, axis=-1)
    neg = tf.keras.backend.max((1.0 - y_true) * y_pred, axis=-1)
    return tf.keras.backend.mean(tf.keras.backend.maximum(0.0, neg - pos + 1), axis=-1)


def dice_and_categorical(y_true, y_pred, alpha=1):
    assert 0 <= alpha <= 1
    return (alpha * dice_coef_loss(y_true, y_pred) + (1 - alpha) * loss(y_true, y_pred)) / 2


def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    n_classes = tf.keras.backend.shape(y_pred)[3]
    y_true = tf.keras.backend.squeeze(y_true, axis=-1)
    y_true = tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int64'), n_classes)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    loss = - y_true * (alpha * tf.keras.backend.pow((1 - y_pred), gamma) * tf.keras.backend.log(y_pred))
    return tf.keras.backend.mean(loss) * 100


class Loss(enum.Enum):
    CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    JACCARD_LOSS = 'jaccard'
    DICE_LOSS = 'dice'
    CATEGORICAL_HINGE = 'categorical_hinge'
    CATEGORCAL_FOCAL = 'categorical_focal'
    DICE_AND_CROSSENTROPY = 'dice_and_crossentropy'

    def __call__(self, *args, **kwargs):
        return {
            Loss.CATEGORICAL_CROSSENTROPY: loss,
            Loss.JACCARD_LOSS: jacard_coef_loss,
            Loss.DICE_LOSS: dice_coef_loss,
            Loss.CATEGORICAL_HINGE: categorical_hinge,
            Loss.CATEGORCAL_FOCAL: categorical_focal_loss,
            Loss.DICE_AND_CROSSENTROPY: dice_and_categorical,
        }[self]


class Monitor(enum.Enum):
    VAL_LOSS = 'val_loss'
    VAL_ACCURACY = 'val_accuracy'
    ACCURACY = 'accuracy'
    LOSS = 'loss'
    DICE_COEF = 'dice_coef'
    JACRAD_COEF = 'jacard_coef'
    FGPA = 'fgpa'
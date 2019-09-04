import tensorflow as tf


def loss(n_classes: int):
    def _loss(y_true, y_pred):
        y_true = tf.keras.backend.reshape(y_true, (-1,))
        y_pred = tf.keras.backend.reshape(y_pred, (-1, n_classes))
        return tf.keras.backend.mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred,
                                                                                     from_logits=True))

    return _loss


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
            return tf.reduce_sum(img, axis=-1)

        fg_equals = tf.multiply(tf.keras.backend.cast(equals, 'int64'), tf.keras.backend.cast(binary, 'int64'))
        fgpa_correct = count_fg(fg_equals)
        fgpa_total = count_fg(binary)
        single_fgpa = tf.divide(tf.cast(fgpa_correct, tf.float32), tf.cast(fgpa_total, tf.float32))
        fgpa = tf.reduce_mean(single_fgpa)

        return fgpa

    return fgpa_accuracy


def jacard_coef(y_true, y_pred):
    n_classes = tf.keras.backend.shape(y_pred)[3]
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = tf.keras.backend.squeeze(y_true, axis=-1)
    y_true = tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int64'), n_classes)

    intersection = tf.reduce_sum(tf.keras.backend.abs(y_true * y_pred), axis=(1, 2))
    sum_ = tf.reduce_sum(tf.keras.backend.abs(y_true + y_pred), axis=(1, 2))
    jac = (intersection + 100) / (sum_ - intersection + 100)
    return tf.reduce_mean(jac, axis=0)


def jacard_coef_loss(y_true, y_pred):
    return -tf.math.log(jacard_coef(y_true, y_pred))


def dice_coef(y_true, y_pred):
    n_classes = tf.keras.backend.shape(y_pred)[3]
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = tf.keras.backend.squeeze(y_true, axis=-1)
    y_true = tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int64'), n_classes)

    intersection = tf.reduce_sum(tf.keras.backend.abs(y_true * y_pred), axis=(1, 2))
    sum_ = tf.reduce_sum(tf.keras.backend.abs(y_true + y_pred), axis=(1, 2))
    dice = (2.0 * intersection + 100) / (sum_ + 100)
    return tf.reduce_mean(dice, axis=0)


def dice_coef_loss(y_true, y_pred):
    return -tf.math.log(dice_coef(y_true, y_pred))

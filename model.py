import tensorflow as tf


def model(input, n_classes):
    # input = tf.image.resize_bilinear(input, size)

    # encoder
    conv1 = tf.layers.conv2d(input, 40, (5, 5), padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 60, (5, 5), padding="same", activation=None)
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding="same")
    conv3 = tf.layers.conv2d(pool2, 120, (5, 5), padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, 120, (5, 5), padding="same", activation=None)
    pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding="same")
    conv5 = tf.layers.conv2d(pool4, 180, (5, 5), padding="same", activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(conv5, 180, (5, 5), padding="same", activation=None)
    pool6 = tf.layers.max_pooling2d(conv6, (2, 2), (2, 2), padding="same")
    conv7 = tf.layers.conv2d(pool6, 240, (5, 5), padding="same", activation=tf.nn.relu)

    # decoder
    deconv1 = tf.layers.conv2d_transpose(conv7, 240, (5, 5), padding="same", activation=tf.nn.relu)
    deconv2 = tf.layers.conv2d_transpose(deconv1, 180, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)
    deconv3 = tf.layers.conv2d_transpose(deconv2, 120, (5, 5), padding="same", activation=tf.nn.relu)
    deconv4 = tf.layers.conv2d_transpose(deconv3, 60, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)
    deconv5 = tf.layers.conv2d_transpose(deconv4, 40, (2, 2), padding="same", strides=(2, 2), activation=None)

    # prediction
    logits = tf.layers.conv2d(deconv5, n_classes, (1, 1), (1, 1), name="logits")
    probs = tf.nn.softmax(logits, -1, name="probabilities")
    prediction = tf.argmax(logits, axis=-1, name="prediction")

    return prediction, logits, probs

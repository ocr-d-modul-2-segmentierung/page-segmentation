import tensorflow as tf


def model(input):
    # input = tf.image.resize_bilinear(input, size)

    # encoder
    conv1 = tf.layers.conv2d(input, 40, (5, 5), padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 60, (5, 5), padding="same", activation=None)
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding="same")
    conv3 = tf.layers.conv2d(pool2, 120, (5, 5), padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, 160, (5, 5), padding="same", activation=None)
    pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding="same")
    conv5 = tf.layers.conv2d(pool4, 240, (5, 5), padding="same", activation=tf.nn.relu)

    # decoder
    deconv1 = tf.layers.conv2d_transpose(conv5, 240, (5, 5), padding="same", activation=tf.nn.relu)
    deconv2 = tf.layers.conv2d_transpose(deconv1, 120, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)
    #deconv2_1 = tf.layers.conv2d_transpose(deconv2, 120, (5, 5), padding="same", activation=tf.nn.relu)
    deconv3 = tf.layers.conv2d_transpose(deconv2, 60, (2, 2), padding="same", strides=(2, 2), activation=None)

    # prediction
    logits = tf.layers.conv2d(deconv3, 9, (1, 1), (1, 1))
    probs = tf.nn.softmax(logits, -1)
    prediction = tf.argmax(logits, axis=-1)

    return prediction, logits, probs

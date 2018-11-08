import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
from lib.md_lstm import *


def model(input, n_classes):
    return model_FCN_skip(input, n_classes)


def model_FCN_skip(input, n_classes):
    # input = tf.image.resize_bilinear(input, size)

    # encoder
    conv1 = tf.layers.conv2d(input, 20, (5, 5), padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 30, (5, 5), padding="same", activation=None)
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding="same")
    conv3 = tf.layers.conv2d(pool2, 40, (5, 5), padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, 40, (5, 5), padding="same", activation=None)
    pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding="same")
    conv5 = tf.layers.conv2d(pool4, 60, (5, 5), padding="same", activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(conv5, 60, (5, 5), padding="same", activation=None)
    pool6 = tf.layers.max_pooling2d(conv6, (2, 2), (2, 2), padding="same")
    conv7 = tf.layers.conv2d(pool6, 80, (5, 5), padding="same", activation=tf.nn.relu)

    # decoder
    deconv1 = tf.layers.conv2d_transpose(conv7, 80, (5, 5), padding="same", activation=tf.nn.relu)
    deconv2 = tf.layers.conv2d_transpose(deconv1, 60, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)
    deconv2 = tf.concat([deconv2, conv6], axis=-1)
    deconv3 = tf.layers.conv2d_transpose(deconv2, 40, (5, 5), padding="same", activation=tf.nn.relu)
    deconv3 = tf.concat([deconv3, conv5], axis=-1)
    deconv4 = tf.layers.conv2d_transpose(deconv3, 30, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)
    deconv4 = tf.concat([deconv4, conv3], axis=-1)
    deconv5 = tf.layers.conv2d_transpose(deconv4, 20, (2, 2), padding="same", strides=(2, 2), activation=None)
    deconv5 = tf.concat([deconv5, conv2], axis=-1)

    # prediction
    logits = tf.layers.conv2d(deconv5, n_classes, (1, 1), (1, 1), name="logits")

    return logits


def model_FCN(input, n_classes):
    # input = tf.image.resize_bilinear(input, size)

    # encoder
    conv1 = tf.layers.conv2d(input, 20, (5, 5), padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 30, (5, 5), padding="same", activation=None)
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding="same")
    conv3 = tf.layers.conv2d(pool2, 40, (5, 5), padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, 40, (5, 5), padding="same", activation=None)
    pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding="same")
    conv5 = tf.layers.conv2d(pool4, 60, (5, 5), padding="same", activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(conv5, 60, (5, 5), padding="same", activation=None)
    pool6 = tf.layers.max_pooling2d(conv6, (2, 2), (2, 2), padding="same")
    conv7 = tf.layers.conv2d(pool6, 80, (5, 5), padding="same", activation=tf.nn.relu)

    # decoder
    deconv1 = tf.layers.conv2d_transpose(conv7, 80, (5, 5), padding="same", activation=tf.nn.relu)
    deconv2 = tf.layers.conv2d_transpose(deconv1, 60, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)
    deconv3 = tf.layers.conv2d_transpose(deconv2, 40, (5, 5), padding="same", activation=tf.nn.relu)
    deconv4 = tf.layers.conv2d_transpose(deconv3, 30, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)
    deconv5 = tf.layers.conv2d_transpose(deconv4, 20, (2, 2), padding="same", strides=(2, 2), activation=None)

    # prediction
    upscaled = tf.image.resize_images(deconv5, tf.shape(input)[1:3])
    logits = tf.layers.conv2d(upscaled, n_classes, (1, 1), (1, 1), name="logits")

    return logits


def get_lstm_cell(num_hidden, reuse_variables=False):
    return cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden, reuse=reuse_variables)


def model_2dlstm(input, n_classes):
    # input = tf.image.resize_bilinear(input, size)

    # encoder
    conv1 = tf.layers.conv2d(input, 16, (5, 5), padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding="same")
    conv2 = tf.layers.conv2d(pool1, 32, (5, 5), padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding="same")
    conv3 = tf.layers.conv2d(pool2, 64, (5, 5), padding="same", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding="same")

    with tf.variable_scope("h"):
        rnn_out = horizontal_standard_lstm(rnn_size=64, input_data=pool3)
    with tf.variable_scope("w"):
        rnn_out = horizontal_standard_lstm(rnn_size=64, input_data=tf.transpose(rnn_out, [0, 2, 1, 3]))

    # decoder
    upscaled = tf.image.resize_images(rnn_out, tf.shape(input)[1:3])

    # prediction
    logits = tf.layers.conv2d(upscaled, n_classes, (1, 1), (1, 1), name="logits")
    probs = tf.nn.softmax(logits, -1, name="probabilities")
    prediction = tf.argmax(logits, axis=-1, name="prediction")

    return prediction, logits, probs
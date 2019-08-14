from typing import Callable, Union, List, Tuple
from pagesegmentation.lib.layers import GraytoRgb

import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
from tensorflow.python.framework.ops import Tensor

Tensors = Union[Tensor, List[Tensor]]


def model(input, n_classes):
    return model_fcn_skip(input, n_classes)


def model_by_name(name: str):
    return {
        "fcn_skip": model_fcn_skip,
        "default": model_fcn_skip,
        "fcn": model_fcn,
        "rest_net": rest_net_fcn,
        "mobile_net": unet_with_mobile_net_encoder,
        "unet": unet,
        "ResUNet": ResUNet,
    }[name]


def calculate_padding(input, scaling_factor: int = 32):
    def scale(i: int, f: int) -> int:
        return (f - i % f) % f

    shape = tf.shape(input)
    px = scale(tf.gather(shape, 1), scaling_factor)
    py = scale(tf.gather(shape, 2), scaling_factor)
    return px, py


def pad(input_tensors):
    input, padding = input_tensors[0], input_tensors[1]
    px, py = padding
    shape = tf.keras.backend.shape(input)
    output = tf.image.pad_to_bounding_box(input, 0, 0, tf.keras.backend.gather(shape, 1) + px,
                                          tf.keras.backend.gather(shape, 2) + py)
    return output


def crop(input_tensors):
    input, padding = input_tensors[0], input_tensors[1]

    if input is None:
        return None

    three_dims = len(input.get_shape()) == 3
    if three_dims:
        input = tf.expand_dims(input, axis=-1)

    px, py = padding
    shape = tf.shape(input)
    output = tf.image.crop_to_bounding_box(input, 0, 0, tf.gather(shape, 1) - px, tf.gather(shape, 2) - py)
    return output


def model_fcn_skip(input: Tensors, n_classes: int):
    input_image = input[0]
    input_binary = input[1]

    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)
    padded = tf.keras.layers.Lambda(pad)([input_image, padding])
    conv1 = tf.keras.layers.Conv2D(20, (5, 5), padding="same", activation=tf.nn.relu,
                                   data_format="channels_last")(padded)
    conv2 = tf.keras.layers.Conv2D(30, (5, 5), padding="same", activation=None,
                                   data_format="channels_last")(conv1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")(conv2)
    conv3 = tf.keras.layers.Conv2D(40, (5, 5), padding="same", activation=tf.nn.relu,
                                   data_format="channels_last")(pool2)
    conv4 = tf.keras.layers.Conv2D(40, (5, 5), padding="same", activation=None,
                                   data_format="channels_last")(conv3)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")(conv4)
    conv5 = tf.keras.layers.Conv2D(60, (5, 5), padding="same", activation=tf.nn.relu,
                                   data_format="channels_last")(pool4)
    conv6 = tf.keras.layers.Conv2D(60, (5, 5), padding="same", activation=None,
                                   data_format="channels_last")(conv5)
    pool6 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")(conv6)
    conv7 = tf.keras.layers.Conv2D(80, (5, 5), padding="same", activation=tf.nn.relu,
                                   data_format="channels_last")(pool6)

    # decoder
    deconv1 = tf.keras.layers.Conv2DTranspose(80, (5, 5), padding="same", activation=tf.nn.relu,
                                              data_format="channels_last")(conv7)
    deconv2 = tf.keras.layers.Conv2DTranspose(60, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu,
                                              data_format="channels_last")(deconv1)
    deconv2 = tf.keras.layers.Concatenate(axis=-1)([deconv2, conv6])

    deconv3 = tf.keras.layers.Conv2DTranspose(40, (5, 5), padding="same", activation=tf.nn.relu,
                                              data_format="channels_last")(deconv2)
    deconv3 = tf.keras.layers.Concatenate(axis=-1)([deconv3, conv5])

    deconv4 = tf.keras.layers.Conv2DTranspose(30, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu,
                                              data_format="channels_last")(deconv3)
    deconv4 = tf.keras.layers.Concatenate(axis=-1)([deconv4, conv3])

    deconv5 = tf.keras.layers.Conv2DTranspose(20, (2, 2), padding="same", strides=(2, 2), activation=None,
                                              data_format="channels_last")(deconv4)
    deconv5 = tf.keras.layers.Concatenate(axis=-1)([deconv5, conv2])
    deconv5 = tf.keras.layers.Lambda(crop)([deconv5, padding])
    # prediction
    logits = tf.keras.layers.Conv2D(n_classes, (1, 1), (1, 1), name="logits")(deconv5)

    model_k = tf.keras.models.Model(inputs=[input_image, input_binary], outputs=logits)

    return model_k


def unet_with_mobile_net_encoder(input: Tensors, n_classes:int):
    input_image = input[0]
    input_binary = input[1]
    rgb_input = GraytoRgb()(input_image)

    rgb_input = tf.keras.applications.mobilenet_v2.preprocess_input(rgb_input * 255)
    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)

    padded = tf.keras.layers.Lambda(pad)([rgb_input, padding])

    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=padded)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = True

    up_stack = [
        tf.keras.layers.Conv2DTranspose(512, 3, strides=2,
                                        padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2DTranspose(256, 3, strides=2,
                                        padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2DTranspose(128, 3, strides=2,
                                        padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2DTranspose(64, 3, strides=2,
                                        padding='same', activation=tf.nn.relu)
    ]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    #down_stack.trainable = False
    x = input
    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = tf.keras.layers.Conv2DTranspose(60, 3, strides=2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Lambda(crop)([x, padding])
    x = tf.keras.layers.Convolution2D(n_classes, 1, 1, name='pred_32', padding='valid')(x)

    return tf.keras.Model(inputs=input, outputs=x)


def rest_net_fcn(input: Tensors, n_classes:int):
    input_image = input[0]
    input_binary = input[1]
    input_image = GraytoRgb()(input_image)
    input_image = tf.keras.applications.resnet50.preprocess_input(input_image * 255)

    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)
    padded = tf.keras.layers.Lambda(pad)([input_image, padding])

    #image_size = tf.keras.backend.int_shape(padded)
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=padded)
    layer_names = [
        'activation_9',  # 256
        'activation_21',  # 512
        'activation_39',  # 1024
        'activation_48',  #2048
        'block_16_project',
    ]
    '''

    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = True

    up_stack = [
        tf.keras.layers.Conv2DTranspose(512, 3, strides=2,
                                        padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2DTranspose(256, 3, strides=2,
                                        padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2DTranspose(128, 3, strides=2,
                                        padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2DTranspose(64, 3, strides=2,
                                        padding='same', activation=tf.nn.relu)
    ]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    
    '''

    x = base_model.get_layer('activation_48').output
    x = tf.keras.layers.Dropout(0.5)(x)
    base_model.trainable = True
    # Todo add skip connections
    x = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)(x)

    x = tf.keras.layers.Lambda(crop)([x, padding])

    x = tf.keras.layers.Convolution2D(n_classes, 1, 1, name='pred_32', padding='valid')(x)

    model_k = tf.keras.models.Model(inputs=input, outputs=x)

    '''
    # allow only some layers to retrain
    train_layers = ['pred_32',
                    'pred_32s'

                    'bn5c_branch2c',
                    'res5c_branch2c',
                    'bn5c_branch2b',
                    'res5c_branch2b',
                    'bn5c_branch2a',
                    'res5c_branch2a',

                    'bn5b_branch2c',
                    'res5b_branch2c',
                    'bn5b_branch2b',
                    'res5b_branch2b',
                    'bn5b_branch2a',
                    'res5b_branch2a',

                    'bn5a_branch2c',
                    'res5a_branch2c',
                    'bn5a_branch2b',
                    'res5a_branch2b',
                    'bn5a_branch2a',
                    'res5a_branch2a']

    for l in model_k.layers:
        if l.name in train_layers:
            l.trainable = True
        else:
            l.trainable = False
    '''
    return model_k


def unet(input: Tensors, n_classes:int):
    input_image = input[0]
    input_binary = input[1]
    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)
    padded = tf.keras.layers.Lambda(pad)([input_image, padding])

    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(padded)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.Lambda(crop)([conv9, padding])

    logits = tf.keras.layers.Convolution2D(n_classes, 1, 1, name='pred_32', padding='valid')(conv9)

    model = tf.keras.models.Model(inputs=[input_image, input_binary], outputs=logits)

    return model


def model_fcn(input: Tensors, n_classes: int):

    # encoder
    input_image = input[0]
    input_binary = input[1]
    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)
    padded = tf.keras.layers.Lambda(pad)([input_image, padding])
    conv1 = tf.keras.layers.Conv2D(20, (5, 5), padding="same", activation=tf.nn.relu)(padded)
    conv2 = tf.keras.layers.Conv2D(30, (5, 5), padding="same", activation=None)(conv1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")(conv2)
    conv3 = tf.keras.layers.Conv2D(40, (5, 5), padding="same", activation=tf.nn.relu)(pool2)
    conv4 = tf.keras.layers.Conv2D(40, (5, 5), padding="same", activation=None)(conv3)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")(conv4)
    conv5 = tf.keras.layers.Conv2D(60, (5, 5), padding="same", activation=tf.nn.relu)(pool4)
    conv6 = tf.keras.layers.Conv2D(60, (5, 5), padding="same", activation=None)(conv5)
    pool6 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")(conv6)
    conv7 = tf.keras.layers.Conv2D(80, (5, 5), padding="same", activation=tf.nn.relu)(pool6)

    # decoder
    deconv1 = tf.keras.layers.Conv2DTranspose(80, (5, 5), padding="same", activation=tf.nn.relu)(conv7)
    deconv2 = tf.keras.layers.Conv2DTranspose(60, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)(deconv1)
    deconv3 = tf.keras.layers.Conv2DTranspose(40, (5, 5), padding="same", activation=tf.nn.relu)(deconv2)
    deconv4 = tf.keras.layers.Conv2DTranspose(30, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)(deconv3)
    deconv5 = tf.keras.layers.Conv2DTranspose(20, (2, 2), padding="same", strides=(2, 2), activation=None)(deconv4)
    deconv5 = tf.keras.layers.Lambda(crop)([deconv5, padding])
    # prediction
    #upscaled = tf.image.resize_images(deconv5, tf.shape(input)[1:3])
    logits = tf.keras.layers.Conv2D(n_classes, (1, 1), (1, 1), name="logits")(deconv5)
    model = tf.keras.models.Model(inputs=[input_image, input_binary], outputs=logits)

    return model


def get_lstm_cell(num_hidden, reuse_variables=False):
    return cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden, reuse=reuse_variables)


def model_2dlstm(input: Tensors, n_classes: int) -> Tuple[Tensor, Tensor, Tensor]:
    # input = tf.image.resize_bilinear(input, size)

    # encoder
    conv1 = tf.keras.layers.Conv2D(16, (5, 5), padding="same", activation=tf.nn.relu)(input)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")(conv1)
    conv2 = tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation=tf.nn.relu)(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(conv2, (2, 2), (2, 2), padding="same")
    conv3 = tf.keras.layers.Conv2D(pool2, 64, (5, 5), padding="same", activation=tf.nn.relu)
    pool3 = tf.keras.layers.MaxPooling2D(conv3, (2, 2), (2, 2), padding="same")

    with tf.variable_scope("h"):
        rnn_out = horizontal_standard_lstm(rnn_size=64, input_data=pool3)
    with tf.variable_scope("w"):
        rnn_out = horizontal_standard_lstm(rnn_size=64, input_data=tf.transpose(rnn_out, [0, 2, 1, 3]))

    # decoder
    upscaled = tf.image.resize_images(rnn_out, tf.shape(input)[1:3])

    # prediction
    logits: Tensor = tf.layers.conv2d(upscaled, n_classes, (1, 1), (1, 1), name="logits")
    probs: Tensor = tf.nn.softmax(logits, -1, name="probabilities")
    prediction: Tensor = tf.argmax(logits, axis=-1, name="prediction")

    return prediction, logits, probs


def ResUNet(input: Tensors, n_classes: int):
    def upsample_concat_block(x, xskip):
        u = tf.keras.layers.UpSampling2D((2, 2))(x)
        c = tf.keras.layers.Concatenate()([u, xskip])
        return c

    def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
        res = conv_block(x, filters, kernel_size, padding, strides)
        res = conv_block(res, filters, kernel_size, padding, 1)
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)
        output = tf.keras.layers.Add()([shortcut, res])
        return output

    def stem(x, filters, kernel_size=3, padding='same', strides=1):
        conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size, padding, strides)
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)
        output = tf.keras.layers.Add()([conv, shortcut])
        return output

    def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
        'convolutional layer which always uses the batch normalization layer'
        conv = bn_act(x)
        conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def bn_act(x, act=True):
        'batch normalization layer with an optinal activation layer'
        x = tf.keras.layers.BatchNormalization()(x)
        if act == True:
            x = tf.keras.layers.Activation('relu')(x)
        return x

    f = [16, 32, 64, 128, 256]
    input_image = input[0]
    input_binary = input[1]
    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)
    padded = tf.keras.layers.Lambda(pad)([input_image, padding])
    ## Encoder
    e0 = padded
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), padding="valid", name="logits")(d4)
    outputs = tf.keras.layers.Lambda(crop)([outputs, padding])

    model = tf.keras.models.Model(input, outputs)
    return model

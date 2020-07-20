import enum
from functools import partial
from typing import Union, List

import efficientnet.tfkeras as efn
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

Tensors = Union[Tensor, List[Tensor]]


def calculate_padding(input, scaling_factor: int = 32):
    def scale(i: int, f: int) -> int:
        return (f - i % f) % f

    shape = tf.shape(input=input)
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
    shape = tf.shape(input=input)
    output = tf.image.crop_to_bounding_box(input, 0, 0, tf.gather(shape, 1) - px, tf.gather(shape, 2) - py)
    return output

def model_fcn_skip(input: Tensors, n_classes: int):
    input_image = input[0]
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

    model_k = tf.keras.models.Model(inputs=input, outputs=logits, name='fcn_skip')

    return model_k


def unet_with_mobile_net_encoder(input: Tensors, n_classes: int):
    input_image = input[0]
    # preprocess to default mobile net input
    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)
    padded = tf.keras.layers.Lambda(pad)([input_image, padding])

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

    # down_stack.trainable = False
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
    x = tf.keras.layers.Convolution2D(n_classes, 1, 1, name='logits', padding='valid')(x)

    return tf.keras.Model(inputs=input, outputs=x, name='mobile_net')


def unet(input: Tensors, n_classes: int):
    input_image = input[0]
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

    logits = tf.keras.layers.Convolution2D(n_classes, 1, 1, name='logits', padding='valid')(conv9)

    model = tf.keras.models.Model(inputs=input, outputs=logits, name='unet')

    return model


def model_fcn(input: Tensors, n_classes: int):
    # encoder
    input_image = input[0]
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
    deconv2 = tf.keras.layers.Conv2DTranspose(60, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)(
        deconv1)
    deconv3 = tf.keras.layers.Conv2DTranspose(40, (5, 5), padding="same", activation=tf.nn.relu)(deconv2)
    deconv4 = tf.keras.layers.Conv2DTranspose(30, (2, 2), padding="same", strides=(2, 2), activation=tf.nn.relu)(
        deconv3)
    deconv5 = tf.keras.layers.Conv2DTranspose(20, (2, 2), padding="same", strides=(2, 2), activation=None)(deconv4)
    deconv5 = tf.keras.layers.Lambda(crop)([deconv5, padding])
    logits = tf.keras.layers.Conv2D(n_classes, (1, 1), (1, 1), name="logits")(deconv5)
    model = tf.keras.models.Model(inputs=input, outputs=logits, name='fcn')

    return model


def res_unet(input: Tensors, n_classes: int):
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

    def bn_act(x, act=True, batch_normailzation=False):
        'batch normalization layer with an optinal activation layer'
        if batch_normailzation:
            x = tf.keras.layers.BatchNormalization()(x)
        if act == True:
            x = tf.keras.layers.Activation('relu')(x)
        return x

    f = [16, 32, 64, 128, 256]
    f = [x * 2 for x in f]
    input_image = input[0]
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
    d4 = tf.keras.layers.Lambda(crop)([d4, padding])

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), padding="valid", name="logits")(d4)

    model = tf.keras.models.Model(input, outputs, name='res_unet')
    return model


def res_net_fine_tuning(input: Tensors, n_classes: int):
    def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1), batch_nm=False):
        conv = tf.keras.layers.Conv2D(filters, (3, 3),
                                      padding="same", kernel_initializer="he_normal",
                                      strides=strides, name=prefix + "_conv")(prevlayer)
        if batch_nm:
            conv = tf.keras.layers.BatchNormalization(name=prefix + "_bn")(conv)
        conv = tf.keras.layers.Activation('relu', name=prefix + "_activation")(conv)
        return conv

    input_image = input[0]

    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x), name='lambda_calc_pad')(input_image)
    padded = tf.keras.layers.Lambda(pad, name='lambda_pad')([input_image, padding])

    # encoder
    resnet_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                 input_tensor=padded)  # (256, 256, 3))

    for l in resnet_base.layers:
        l.trainable = True
    # skips
    conv1 = resnet_base.get_layer("conv1_relu").output
    conv2 = resnet_base.get_layer("conv2_block3_out").output
    conv3 = resnet_base.get_layer("conv3_block4_out").output
    conv4 = resnet_base.get_layer("conv4_block6_out").output
    conv5 = resnet_base.get_layer("conv5_block3_out").output

    # bridge
    conv5 = conv_block_simple(conv5, 256, "b_1")

    # decoder
    up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    up10 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv9), padded], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    outc = tf.keras.layers.Lambda(crop, name='lambdacrop')([conv10, padding])
    out = tf.keras.layers.Convolution2D(n_classes, 1, 1, name='logits', padding='valid')(outc)

    model = tf.keras.Model(input, out, name='image_res_net')
    return model


def eff_net_fine_tuning(input: Tensors, n_classes: int, efnet=efn.EfficientNetB1):
    def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1), batch_nm=False):
        conv = tf.keras.layers.Conv2D(filters, (3, 3),
                                      padding="same", kernel_initializer="he_normal",
                                      strides=strides, name=prefix + "_conv")(prevlayer)
        if batch_nm:
            conv = tf.keras.layers.BatchNormalization(name=prefix + "_bn")(conv)
        conv = tf.keras.layers.Activation('relu', name=prefix + "_activation")(conv)
        return conv

    input_image = input[0]
    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)
    padded = tf.keras.layers.Lambda(pad)([input_image, padding])

    # encoder
    effnet_base = efnet(weights='imagenet', include_top=False, input_tensor=padded)  # input_shape=(256, 256, 3))
    for l in effnet_base.layers:
        l.trainable = True

    # skips
    conv1 = effnet_base.get_layer("block2a_expand_activation").output
    conv2 = effnet_base.get_layer("block3a_expand_activation").output
    conv3 = effnet_base.get_layer("block4a_expand_activation").output
    conv4 = effnet_base.get_layer("block6a_expand_activation").output

    # bridge
    conv4 = conv_block_simple(conv4, 256, "b_1")

    # decoder
    up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up6, 256, "conv6_1")
    conv5 = conv_block_simple(conv5, 256, "conv6_2")

    up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up7, 196, "conv7_1")
    conv6 = conv_block_simple(conv6, 196, "conv7_2")

    up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv6), conv1], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv8), padded], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    out = tf.keras.layers.Lambda(crop)([conv9, padding])
    out = tf.keras.layers.Convolution2D(n_classes, 1, 1, name='logits', padding='valid')(out)

    model = tf.keras.Model(input, out, name='effb0')
    return model


def default_preprocess(x):
    return x / 255.0


class Architecture(enum.Enum):
    FCN_SKIP = 'fcn_skip'
    FCN = 'fcn'
    RES_NET = 'image_res_net'
    RES_UNET = 'res_unet'
    MOBILE_NET = 'mobile_net'
    UNET = 'unet'
    EFFNETB0 = 'effb0'
    EFFNETB1 = 'effb1'
    EFFNETB2 = 'effb2'
    EFFNETB3 = 'effb3'
    EFFNETB4 = 'effb4'
    EFFNETB5 = 'effb5'
    EFFNETB6 = 'effb6'
    EFFNETB7 = 'effb7'

    def __call__(self, *args, **kwargs):
        return self.model()

    def model(self):
        return {
            Architecture.FCN_SKIP: model_fcn_skip,
            Architecture.FCN: model_fcn,
            Architecture.RES_NET: res_net_fine_tuning,
            Architecture.RES_UNET: res_unet,
            Architecture.MOBILE_NET: unet_with_mobile_net_encoder,
            Architecture.UNET: unet,
            Architecture.EFFNETB0: partial(eff_net_fine_tuning, efnet=efn.EfficientNetB0),
            Architecture.EFFNETB1: partial(eff_net_fine_tuning, efnet=efn.EfficientNetB1),
            Architecture.EFFNETB2: partial(eff_net_fine_tuning, efnet=efn.EfficientNetB2),
            Architecture.EFFNETB3: partial(eff_net_fine_tuning, efnet=efn.EfficientNetB3),
            Architecture.EFFNETB4: partial(eff_net_fine_tuning, efnet=efn.EfficientNetB4),
            Architecture.EFFNETB5: partial(eff_net_fine_tuning, efnet=efn.EfficientNetB5),
            Architecture.EFFNETB6: partial(eff_net_fine_tuning, efnet=efn.EfficientNetB6),
            Architecture.EFFNETB7: partial(eff_net_fine_tuning, efnet=efn.EfficientNetB7),
        }[self]

    def preprocess(self):
        return {
            Architecture.FCN_SKIP: (default_preprocess, False),
            Architecture.FCN: (default_preprocess, False),
            Architecture.RES_NET: (tf.keras.applications.resnet50.preprocess_input, True),
            Architecture.RES_UNET: (default_preprocess, False),
            Architecture.MOBILE_NET: (tf.keras.applications.mobilenet_v2.preprocess_input, True),
            Architecture.UNET: (default_preprocess, False),
            Architecture.EFFNETB0: (efn.preprocess_input, True),
            Architecture.EFFNETB1: (efn.preprocess_input, True),
            Architecture.EFFNETB2: (efn.preprocess_input, True),
            Architecture.EFFNETB3: (efn.preprocess_input, True),
            Architecture.EFFNETB4: (efn.preprocess_input, True),
            Architecture.EFFNETB5: (efn.preprocess_input, True),
            Architecture.EFFNETB6: (efn.preprocess_input, True),
            Architecture.EFFNETB7: (efn.preprocess_input, True),
        }[self]


class Optimizers(enum.Enum):
    ADAM = 'adam'
    ADAMAX = 'adamax'
    ADADELTA = 'adadelta'
    ADAGRAD = 'adagrad'
    RMSPROP = 'rmsprop'
    SGD = 'sgd'
    NADAM = 'nadam'

    def __call__(self, *args, **kwargs):
        return {
            Optimizers.ADAM: tf.keras.optimizers.Adam,
            Optimizers.ADAMAX: tf.keras.optimizers.Adamax,
            Optimizers.ADADELTA: tf.keras.optimizers.Adadelta,
            Optimizers.ADAGRAD: tf.keras.optimizers.Adagrad,
            Optimizers.RMSPROP: tf.keras.optimizers.RMSprop,
            Optimizers.SGD: tf.keras.optimizers.SGD,
            Optimizers.NADAM: tf.keras.optimizers.Nadam,
        }[self]


if __name__ == '__main__':
    #effnet_base = efn.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    #print(effnet_base.summary())
    pass
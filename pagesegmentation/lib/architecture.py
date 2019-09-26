import segmentation_models as sm
from typing import Callable, Union, List, Tuple
from pagesegmentation.lib.layers import GraytoRgb
from tensorflow.python.framework.ops import Tensor
from segmentation_models import get_preprocessing

import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
import enum

Tensors = Union[Tensor, List[Tensor]]


class ModelStructure(enum.Enum):
    UNET = 'unet'
    FPN = 'fpn'
    LINKNET = 'linknet'
    PSPNET = 'pspnet'

    def __call__(self, *args, **kwargs):
        return {
            ModelStructure.UNET: sm.Unet,
            ModelStructure.FPN: sm.FPN,
            ModelStructure.LINKNET: sm.Linknet,
            ModelStructure.PSPNET: sm.PSPNet,

        }[self]


class DecoderBlockType(enum.Enum):
    UPSAMPLING = 'upsampling'
    TRANSPOSE = 'transpose'


class Backbone(enum.Enum):
    VGG16 = 'vgg16'
    VGG19 = 'vgg19'
    ResNet18 = 'resnet18'
    ResNet34 = 'resnet34'
    ResNet50 = 'resnet50'
    ResNet101 = 'resnet101'
    ResNet152 = 'resnet152'
    SEResNet18 = 'seresnet18'
    SEResNet34 = 'seresnet34'
    SEResNet50 = 'seresnet50'
    SEResNet101 = 'seresnet101'
    SEResNet152 = 'seresnet152'
    ResNeXt50 = 'resnext50'
    ResNeXt101 = 'resnext101'
    SeResNeXt50 = 'seresnext50'
    SeResNeXt101 = 'seresnext101'
    SENet154 = 'senet154'
    DenseNet121 = 'densenet121'
    DenseNet169 = 'densenet169'
    DenseNet201 = 'densenet201'
    Inception3 = 'inceptionv3'
    InceptionResNetv2 = 'inceptionresnetv2'
    MobileNet = 'mobilenet'
    MobileNetv2 = 'mobilenetv2'
    EfficientNetb0 = 'efficientnetb0'
    EfficientNetb1 = 'efficientnetb1'
    EfficientNetb2 = 'efficientnetb2'
    EfficientNetb3 = 'efficientnetb3'
    EfficientNetb4 = 'efficientnetb4'
    EfficientNetb5 = 'efficientnetb5'
    EfficientNetb6 = 'efficientnetb6'
    EfficientNetb7 = 'efficientnetb7'


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


def model_factory(input: Tensors, n_classes: int, model_structure: ModelStructure= ModelStructure.UNET,
                  encoder_structure: Backbone = Backbone.ResNet18,
                  #decoder_block_type: DECODERBLOCKTYPE = DECODERBLOCKTYPE.TRANSPOSE,
                  pretrainend: bool = True):

    preprocessing = get_preprocessing(encoder_structure.value)
    input_image = input[0]
    if input_image.shape != 3:
        input_image = GraytoRgb()(input_image)
    input_image = tf.keras.layers.Lambda(lambda x: x * 255)(input_image)

    input_image = tf.keras.layers.Lambda(lambda x: preprocessing(x))(input_image)
    padding = tf.keras.layers.Lambda(lambda x: calculate_padding(x))(input_image)
    padded = tf.keras.layers.Lambda(pad)([input_image, padding])

    if pretrainend:
        base_model = model_structure()(encoder_structure.value, encoder_weights='imagenet', classes=n_classes,
                                       #decoder_block_type=decoder_block_type.value,
                                       activation='relu',
                                       decoder_use_batchnorm=False)(padded)
    else:
        base_model = model_structure()(encoder_structure.value, classes=n_classes,
                                       #decoder_block_type=decoder_block_type.value,
                                       activation='relu',
                                       decoder_use_batchnorm=False)(padded)
    deconv5 = tf.keras.layers.Lambda(crop)([base_model, padding])
    logits = tf.keras.layers.Conv2D(n_classes, (1, 1), (1, 1), name="logits")(deconv5)
    model = tf.keras.models.Model(inputs=input, outputs=logits)
    return model


if __name__ == "__main__":
    from segmentation_models import get_preprocessing
    preprocess_input = get_preprocessing(Backbone.ResNet18.value)
    model_factory(None, 2, ModelStructure.UNET, Backbone.ResNet18)









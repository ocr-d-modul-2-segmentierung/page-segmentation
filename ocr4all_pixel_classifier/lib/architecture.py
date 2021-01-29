import enum
from functools import partial


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
    PROB_UNET = 'prob_unet'

    def __call__(self, *args, **kwargs):
        return self.model()

    def model(self):
        from ocr4all_pixel_classifier.lib.model import model_fcn_skip, model_fcn, res_net_fine_tuning, res_unet, \
            unet_with_mobile_net_encoder, unet, eff_net_fine_tuning
        from efficientnet import tfkeras as efn
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
            Architecture.PROB_UNET: prob_unet,
        }[self]

    def preprocess(self):
        from efficientnet import tfkeras as efn
        import tensorflow as tf

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
            Architecture.PROB_UNET: (default_preprocess, False),
        }[self]


def default_preprocess(x):
    return x / 255.0


class Optimizers(enum.Enum):
    ADAM = 'adam'
    ADAMAX = 'adamax'
    ADADELTA = 'adadelta'
    ADAGRAD = 'adagrad'
    RMSPROP = 'rmsprop'
    SGD = 'sgd'
    NADAM = 'nadam'

    def __call__(self, *args, **kwargs):
        import tensorflow.keras.optimizers as tfo
        return {
            Optimizers.ADAM: tfo.Adam,
            Optimizers.ADAMAX: tfo.Adamax,
            Optimizers.ADADELTA: tfo.Adadelta,
            Optimizers.ADAGRAD: tfo.Adagrad,
            Optimizers.RMSPROP: tfo.RMSprop,
            Optimizers.SGD: tfo.SGD,
            Optimizers.NADAM: tfo.Nadam,
        }[self]

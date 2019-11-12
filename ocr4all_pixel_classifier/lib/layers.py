import tensorflow as tf


class Padding2DTensor(tf.keras.layers.Layer):

    def __init__(self, padding=[1,1],
                 data_format="channels_last", **kwargs):
        self.data_format = data_format
        super(Padding2DTensor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Padding2DTensor, self).build(input_shape)

    def call(self, inputs, mask=None):
        input, padding = inputs[0], inputs[1]
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('Padding2DTensor must be called on a list of tensors '
                            'with size2). Got: ' + str(inputs))
        px, py = padding
        shape = tf.shape(input=input)
        output = tf.image.pad_to_bounding_box(input, 0, 0, tf.gather(shape, 1) + px, tf.gather(shape, 2) + py)
        return output

    def compute_output_shape(self, input_shape):

        shape = tf.keras.backend.shape(input_shape[1])
        return input_shape[0][0], input_shape[0][1] + self.padding[0], input_shape[0][2] + self.padding[1], \
               input_shape[0][3]


class GraytoRgb(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
        # expand your input from gray scale to rgb
        # if your inputs.shape = (None,None,1)
        fake_rgb = tf.keras.backend.concatenate([inputs for i in range(3)], axis=-1)
        fake_rgb = tf.keras.backend.cast(fake_rgb, 'float32')
        # else use K.stack( [inputs for i in range(3)], axis=-1 )
        # preprocess for uint8 image
        return fake_rgb

    def compute_output_shape( self, input_shape):
        return input_shape[:3] + (3,)

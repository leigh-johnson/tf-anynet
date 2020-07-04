import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv3d_block(
    inputs,
    out_channels,
    initializer_cls=keras.initializers.VarianceScaling, 
    padding='same',
    kernel_size=3,     
    dilation_rate=1, 
    batch_norm=True,
    momentum=0.9,
    epsilon=1e-5,
    strides=1
    ):
    
    if batch_norm:
        x = layers.BatchNormalization(
                momentum=momentum,
                epsilon=epsilon,
        )(inputs)
        x = layers.Conv3D(
            out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            activation='relu',
            dilation_rate=dilation_rate,
            kernel_initializer=initializer_cls()
        )(x)
        return x
    else:
        return layers.Conv2D(
            out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation='relu',
            dilation_rate=dilation_rate,
            kernel_initializer=initializer_cls()
        )(inputs)


def conv3d_net(inputs, layers, out_channels):
    net = conv3d_block(inputs, out_channels)
    
    for _ in range(layers):
        net = conv3d_block(net, out_channels)

    return conv3d_block(net, 1)


class Conv3DRegularizer(keras.layers.Layer):

    def __init__(self, nlayers, out_channels):
        super(Conv3DRegularizer, self).__init__()

        self.nlayers = nlayers
        self.out_channels = out_channels
    
    def build(self, input_shapes):

        inputs = layers.Input(shape=input_shapes[1:])
        self.conv3d_net = conv3d_net(inputs, self.nlayers, self.out_channels)
        self.model = keras.Model(inputs, self.conv3d_net)

    def call(self, inputs):
        return self.model(inputs)
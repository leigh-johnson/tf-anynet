import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv2d_block(
    inputs,
    out_channels, 
    kernel_size=3, 
    strides=1, 
    padding='same', 
    initializer_cls=keras.initializers.VarianceScaling, 
    dilation_rate=1, 
    batch_norm=True,
    momentum=0.9,
    epsilon=1e-5,
    ):
    
    if batch_norm:
        x = layers.BatchNormalization(
                momentum=momentum,
                epsilon=epsilon,
        )(inputs)
        x = layers.Conv2D(
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

@keras.utils.register_keras_serializable(package='AnyNet')
class UnetUp2(keras.layers.Layer):
    def __init__(self, out_channels):
        super(UnetUp2, self).__init__()

        #self.in_channels = in_channels
        self.out_channels = out_channels
        #self.batch_size = batch_size

        #inputs = layers.Input(shape=in_channels, batch_size=self.batch_size)


    def build(self, input_shapes):
        input_shape1, input_shape2 = input_shapes

        self.input_layer1 = layers.Input(shape=input_shape1[1:], batch_size=input_shape1[0], name="input_unet_0")
        self.input_layer2 = layers.Input(shape=input_shape2[1:], batch_size=input_shape2[0], name="input_unet_1")
        
        outputs2 = layers.UpSampling2D(
            size=2, interpolation='bilinear',
        )(self.input_layer2)

        bottom, right = self.input_layer1.shape[1]%2, self.input_layer1.shape[2]%2
        padding = (( 0, bottom), (0, right))
        outputs2 = layers.ZeroPadding2D(padding=padding)(outputs2)
        concat = layers.Concatenate(axis=-1)([self.input_layer1, outputs2])

        self.output_layer = conv2d_block(concat, self.out_channels)

        self.model = keras.Model([
            self.input_layer1,
            self.input_layer2],
            self.output_layer
        )

    def call(self, inputs):

        return self.model(inputs)

@keras.utils.register_keras_serializable(package='AnyNet')
class FeatureExtractor(keras.layers.Layer):
    def __init__(self, init_filters, nblocks, batch_size):
        '''Downsample inputs, extract features, upsample extracted features

            Args:
                inputs (tuple): (left_img_data, right_img_data)
                init_filters (int): num initial output filters (increases 2*init_filters per block)
                blocks (int): num (BatchNorm2D -> ReLU -> Conv2D) blocks per resolution

            Returns:
                keras.Model
        '''
        super(FeatureExtractor, self).__init__()
        
        self.initializer_cls = keras.initializers.VarianceScaling
        self.init_filters = init_filters
        self.nblocks = nblocks
        self.batch_size = batch_size

    def get_config(self):
        config = {
            'init_filters': self.init_filters,
            'nblocks': self.nblocks,
            'batch_size': self.batch_size
        }
        config.update(super().get_config())

        return config
    def build(self, input_shape):
 
        self.input_layer = layers.Input(shape=input_shape[1:], batch_size=input_shape[0], name="input_feature_ext")

        self.block0 = self._make_block0(
            self.init_filters,
            self.nblocks,
            self.initializer_cls
        )

            
        blocks = [self.block0]

        ##
        # Connect downsample/upsample block skip connections
        # Fig. 3: U-Net Feature Extractor.
        # {Anytime Stereo Image Depth Estimation on Mobile Devices},
        # {Wang, Yan and Lai, Zihang and Huang, Gao and Wang, Brian H. and Van Der Maaten, Laurens and Campbell, Mark and Weinberger, Kilian Q}
        ##

        # dblock_idxs = range(self.nblocks+1) # n=2 (0,1,2)
        # ublock_idxs = range(self.nblocks+1,(self.nblocks*2)+1) # n=2 (3,4)

        nConnects = self.nblocks*self.init_filters

        # create and connect downsample blocks with block0 input # n=2 0 -> (1 -> 2)
        for i in range(self.nblocks):
            blocks.append(
                self._make_block(
                blocks[-1], # (2**i)*nConnects inputs
                (self.nblocks**(i+1))*nConnects, # (2**(i+1))*nConnects
                self.nblocks)
            )
        
        # create and connect upsample unet blocks with final downsample block # n=2 2 -> (3 -> 4)
        for i,n in enumerate(reversed(range(self.nblocks))):
            unet = UnetUp2(
                # nConnects=2, i=0, n=1 => in_channels == 12
                # nConnects=2, i=0, n=1 => out_channels == 8
                #inputs1=blocks[n], 
                #inputs2=blocks[n+1],
                out_channels=blocks[n+1].shape[-1],
                #batch_size=batch_size
            )
            #import pdb; pdb.set_trace()
            unet = unet([blocks[n], blocks[n+1]])
            blocks.append(unet)

        self.blocks = blocks
        
        self.model = keras.Model(
            inputs=self.input_layer, 
            outputs=[
                blocks[-i]
                for i in
                reversed(list(range(1, self.nblocks+2)))
            ]
        )

    def call(self, inputs):
        return self.model(inputs)


    def _make_block(self, inputs, out_channels, nblocks, padding='valid'):
        x = layers.MaxPooling2D(padding=padding)(inputs)
        for i in range(nblocks):
            x = conv2d_block(
                x,
                out_channels
            )
        return x

    def _make_block0(self, out_channels, nblocks, initializer_cls):
        
        conv_input = layers.Conv2D(
            out_channels, 
            kernel_size=3, 
            strides=1, 
            #padding='same',
            use_bias=False,
            kernel_initializer=initializer_cls(),
        )(self.input_layer)

        downsample_conv = conv2d_block(
            inputs=conv_input,
            out_channels=out_channels,
            strides=2,
         )

        block0 = self._make_block(
            inputs=downsample_conv,
            out_channels=out_channels*2,
            nblocks=nblocks
        )

        return block0

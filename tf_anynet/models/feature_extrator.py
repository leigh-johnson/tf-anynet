import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv3d_block(
    filters, 
    initializer_cls=keras.initializers.VarianceScaling, 
    padding='same',
    kernel_size=3, stride=1, momentum=0.9, epsilon=1e-5):
    x = layers.BatchNorm3D(
        momentum=momentum,
        epsilon=epsilon
    )
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        activation='relu',
        dilation_rate=dilation_rate,
        kernel_initializer=initializer_cls()
    )(x)
    return x

def conv3d_net(layers, filters):
    net = conv3d_block(filters)
    
    for _ in range(layers):
        net = conv3d_block(filters)(net)

    return conv3d_block(1)(net)

def conv2d_block(
    #in_channels,
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
        # x = BatchNorm()(in_channels)
        # x = layers.Conv2D(
        #     filters,
        #     kernel_size=kernel_size,
        #     strides=strides,
        #     padding=padding,
        #     use_bias=False,
        #     activation='relu',
        #     dilation_rate=dilation_rate,
        #     kernel_initializer=initializer_cls()
        # )(x)
        # return x
        return keras.models.Sequential([
            layers.BatchNormalization(
                momentum=momentum,
                epsilon=epsilon,
                #input_shape=in_channels
            ),
            layers.Conv2D(
                out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                activation='relu',
                dilation_rate=dilation_rate,
                kernel_initializer=initializer_cls()
            )
        ])
    else:
       return layers.Conv2D(
            out_channels,
            #input_shape=in_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            activation='relu',
            dilation_rate=dilation_rate,
            kernel_initializer=initializer_cls()
        )

class UnetUp(keras.Model):
    def __init__(self, filters):
        super(UnetUp, self).__init__()

        self.upsample = layers.UpSampling2D(
            size=2, interpolation='bilinear',
        )
        self.conv2d = conv2d_block(filters)(conv2d_block(filters))

    def call(self, inputs1, inputs2):
        outputs2 = self.upsample(inputs2)

        bottom, right = inputs1.shape(1)%2, inputs1.size(2)%2
        padding = (( 0, bottom), (0, right))
        outputs2 = layers.ZeroPadding2D(padding=padding)(outputs2)
        return self.conv2d(
            layers.Concatenate(axis=1)([inputs1, outputs2])
        )



class FeatureExtractor(keras.Model):
    def __init__(self, init_filters, nblocks, input_dim, batch_size):
        '''Downsample inputs, extract features, upsample extracted features

            Args:
                inputs (tuple): (left_img_data, right_img_data)
                init_filters (int): num initial output filters (increases 2*init_filters per block)
                blocks (int): num (BatchNorm2D -> ReLU -> Conv2D) blocks per resolution

            Returns:
                keras.Model
        '''
        super(FeatureExtractor, self).__init__()
        
        initializer_cls = keras.initializers.VarianceScaling
        self.init_filters = init_filters
        self.nblocks = nblocks
        self.input_dim = input_dim
        self.batch_size = batch_size
        # block0 downsamples input through maxpooling op
        # it would also be possible to use strided convolution to downsample here
        
        self.block0 = self._make_block0(
            init_filters,
            nblocks,
            initializer_cls
        )
        
        next_filters = 2 * init_filters
        self.downsample_blocks = []
        for i in range(nblocks):
            self.downsample_blocks.append(self._make_block(
                2**(i+1)*next_filters,
                nblocks
            ))
        
        # 3x3 conv with 8 filters
        # warped fed as input to cost block 1 in cost volume network
        self.cost_volume_input1 = self.downsample_blocks[-1]


        # upsampling blocks
        self.upsample_blocks = []
        for i in reversed(range(nblocks)):
            self.upsample_blocks.append(
                UnetUp(
                next_filters*2**i
                )
            )
        self.cost_volume_input2 = self.upsample_blocks[3]
        self.cost_volume_input3 = self.upsample_blocks[7]

        def call(self, inputs):
            blocks = [self.block0(inputs)]

            # connect downsample block inputs/outputs
            for i in range(self.nblocks):
                blocks.append(
                    self.downsample_blocks[i](blocks[-1])
                )
            
            blocks = list(reversed(blocks))

            # connect upsample block inputs/outputs
            for i in range(1, 3):
                blocks[i] = self.upsample_blocks[i-1](blocks[i], blocks[i-1])
            return blocks
        
        

    def _make_block(self, out_channels, nblocks, padding='valid'):
        model = [layers.MaxPooling2D(padding=padding)]
        for i in range(nblocks):
            model.append(
                conv2d_block(
                    out_channels
            ))
        return keras.models.Sequential(model)

    def _make_block0(self, filters, nblocks, initializer_cls):
        
        input_shape = (self.input_dim[0], self.input_dim[1], 3)
        downsample_input = layers.Conv2D(
            filters, 
            kernel_size=3, 
            strides=1, 
            #padding='same',
            use_bias=False,
            input_shape=input_shape,
            kernel_initializer=initializer_cls(),
        )

        downsample_conv = conv2d_block(
            out_channels=filters,
            strides=2
        )

        block = self._make_block(
            out_channels=filters*2,
            nblocks=nblocks
        )

        downsample_conv = keras.models.Sequential([
            downsample_input,
            downsample_conv,
            block
        ])


        return downsample_conv


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class UnetUp(keras.Model):
    def __init__(self, filters, inputs):
        super(Unet, self).__init__()
        
        conv2d_initialzer = keras.initializers.VarianceScaling()
        # 512x256 input
        # block0 (1 -> 2 filters)
        x = layers.Conv2D(
            filters, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding=(1,1),
            use_bias=False,
            kernel_initializer=conv2d_initialzer
        (inputs))
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            1,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
            use_bias=False,
            activation='relu',
            kernel_initializer=conv2d_initialzer
        )(x)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            2,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu',
            kernel_initializer=conv2d_initialzer
        )
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            2,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )

        # block 1 (2 -> 4 filters)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            4,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )(x)
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            4,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )(x)

        # block 2 (4 -> 8 filters)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            8,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )(x)
        x = layers.BatchNorm2D()(x)

        self.cost_volume_input1 = layers.Conv2D(
            8,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )(x)

        # upsampling blocks
        # upsample by factor of two
        x = layers.UpSampling2D(
            size=2, interpolation='bilinear',
        )(self.cost_volume_input1)

        # 12 -> 4
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            4,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )(x)    
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            4,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )(x)

        # 6 -> 2
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            2,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )(x)    
        x = layers.BatchNorm2D()(x)
        x = layers.Conv2D(
            2,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            use_bias=False,
            activation='relu'
            kernel_initializer=conv2d_initialzer
        )(x)   

    def call(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        buttom, right = inputs1.size(2)%2, inputs1.size(3)%2
        outputs2 = F.pad(outputs2, (0, -right, 0, -buttom))
        return self.conv(torch.cat([inputs1, outputs2], 1))


class AnyNet(keras.Model):

    def __init__(self, 
        unet_conv2d_filters=1,
        unet_blocks=2,
        spn_conv2d_filters=8,
        disp_conv3d_filters=4,
        disp_conv3d_layers=4,
        disp_conv3d_growth_rate=[4, 1, 1],
        local_max_disps=[12, 3, 3],
        global_max_disp=192,
        loss_weights=(0.25, 0.5, 1.0, 1.0),
        learning_rate=5e-4,
        input_dim=(256, 512) # height, weight
        *args, **kwargs
        ):
        '''AnyNet keras implementation

            Args:
                learning_rate (float): Adam optimizer
                loss_weights (tuple): weight multipler per stage e.g. 0.25 weight applied to Unet stage
                local_max_disps (list[int]): Maximum disparity localized per Disparity Net stage
                global_max_disp (int): Global maximum disparity (pixels)
                unet_conv2d_filters (int): Initial num Conv2D output filters of Unet feature extractor
                unet_blocks (int): Num (BatchNorm2D -> ReLU -> Conv2D) blocks per Unet stage
                spn_conv2d_filters (int): Initial num Conv2D output filters of Spatial Propagation Network (SPN)
                disp_conv3d_filters (int): Initial num Conv3D output filters of Disparity Network (multiplied by growth_rate[stage_num])
                disp_conv3d_layers (int): Num Conv3D layers of Disparty Network
            Returns: 
                keras.Model
        '''
        super(AnyNet, self).__init__()

        left_img = keras.Input(shape=(3, input_dim[0], input_dim[1]), name="left_img_input")
        right_img = keras.Input(shape=(3, input_dim[0], input_dim[1]), name="right_img_input")
        self.inputs = [left_img, right_img]

        unet  = UnetUp(self.unet_conv2d_filters, self.inputs)
        
    




 # torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')       


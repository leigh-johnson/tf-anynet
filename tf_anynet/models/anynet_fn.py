
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.image import dense_image_warp

#from .cspn import ConvSpatialPropagationNet
from .dispnet import DisparityNetwork
from .feature_extractor import FeatureExtractor

#@keras.utils.register_keras_serializable(package='AnyNet')
class AnyNet(object): 
    def __init__(self, 
        unet_conv2d_filters=1,
        unet_nblocks=2,
        with_cspn=None,
        cspn_conv3d_filters=8,
        cspn_conv3d_step=24,
        disp_conv3d_filters=4,
        disp_conv3d_layers=4,
        disp_conv3d_growth_rate=[4, 1, 1],
        local_max_disps=[12, 3, 3],
        global_max_disp=192,
        stages=3,
        batch_size=None,
        eval_data=None,
        *args, **kwargs
        ):
        '''AnyNet keras implementation

            Args:
                learning_rate (float): Adam optimizer
                loss_weights (tuple): weight multipler per stage e.g. 0.25 weight applied to Unet stage
                local_max_disps (list[int]): Maximum disparity localized per Disparity Net stage
                global_max_disp (int): Global maximum disparity (pixels)
                unet_conv2d_filters (int): Initial num Conv2D output filters of Unet feature extractor
                unet_nblocks (int): Num (BatchNorm2D -> ReLU -> Conv2D) blocks per Unet stage
                spn_conv2d_filters (int): Initial num Conv2D output filters of Spatial Propagation Network (SPN)
                disp_conv3d_filters (int): Initial num Conv3D output filters of Disparity Network (multiplied by growth_rate[stage_num])
                disp_conv3d_layers (int): Num Conv3D layers of Disparty Network
            Returns: 
                keras.Model
        '''
    
        self.unet_conv2d_filters = unet_conv2d_filters
        self.unet_nblocks = unet_nblocks
        self.disp_conv3d_filters = disp_conv3d_filters
        self.disp_conv3d_layers = disp_conv3d_layers
        self.disp_conv3d_growth_rate = disp_conv3d_growth_rate
        self.cspn_conv3d_filters = cspn_conv3d_filters
        self.with_cspn = with_cspn      
        self.local_max_disps = local_max_disps
        self.global_max_disp = global_max_disp
        self.stages = stages
        self.batch_size = batch_size

        self.feature_extractor  = FeatureExtractor(
            init_filters=unet_conv2d_filters,
            nblocks=unet_nblocks,
            batch_size=batch_size
        )

        self.disparity_network = DisparityNetwork(
            disp_conv3d_filters=disp_conv3d_filters,
            disp_conv3d_layers=disp_conv3d_layers,
            disp_conv3d_growth_rate= disp_conv3d_growth_rate,
            local_max_disps=local_max_disps,
            global_max_disp=global_max_disp,
            stages=stages,
            batch_size=batch_size
        )

        # if with_cspn:
        #     self.cspnet = ConvSpatialPropagationNet(
        #         cspn_conv2d_filters,
        #         cspn_conv2d_step
        #     )

    def build(self, input_shape):
        
        # import pdb; pdb.set_trace()
        left_img = keras.Input(shape=input_shape[1:], name="input_left_img")
        right_img = keras.Input(shape=input_shape[1:], name="input_right_img")


        _, height, width, _ = input_shape
        
        feats_l = self.feature_extractor(left_img)
        feats_r = self.feature_extractor(right_img)

        self.disparity_network.height = height
        self.disparity_network.width = width
        output = self.disparity_network([feats_l, feats_r ])
        return keras.Model([left_img, right_img], {
            f'disparity-{i}': x for i,x in enumerate(output)
        }, name="anynet")
    

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.image import dense_image_warp

from .dispnet import DisparityNetwork
from .feature_extractor import FeatureExtractor

#@keras.utils.register_keras_serializable(package='AnyNet')
class AnyNet(object):
    def __init__(self, 
        unet_conv2d_filters=1,
        unet_nblocks=2,
        spn_conv2d_filters=8,
        disp_conv3d_filters=4,
        disp_conv3d_layers=4,
        disp_conv3d_growth_rate=[4, 1, 1],
        local_max_disps=[12, 3, 3],
        global_max_disp=192,
        loss_weights=(0.25, 0.5, 1.0, 1.0),
        learning_rate=5e-4,
        stages=3,
        batch_size=8,
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
    
        self.loss_weights = loss_weights
        self.unet_conv2d_filters = unet_conv2d_filters
        self.unet_nblocks = unet_nblocks
        self.spn_conv2d_filters = spn_conv2d_filters
        self.disp_conv3d_filters = disp_conv3d_filters
        self.disp_conv3d_layers = disp_conv3d_layers
        self.disp_conv3d_growth_rate = disp_conv3d_growth_rate
        self.local_max_disps = local_max_disps
        self.global_max_disp = global_max_disp
        self.learning_rate = learning_rate
        self.stages = stages
        self.eval_data = eval_data
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
            loss_weights=loss_weights,
            stages=stages,
            batch_size=batch_size
        )

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
        return keras.Model([left_img, right_img], output, name="anynet")
    
    # def get_config(self):
    #     config = {
    #         'init_filters': self.init_filters,
    #         'nblocks': self.nblocks,
    #         'batch_size': self.batch_size
    #     }
    #     return config
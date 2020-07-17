
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.image import dense_image_warp

#from .cspn import ConvSpatialPropagationNet
from .dispnet import DisparityNetworkStageN, DisparityNetworkStage0
from .feature_extractor import FeatureExtractor

@keras.utils.register_keras_serializable(package='AnyNet')
class AnyNetV2(keras.layers.Layer): 
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
        *args, **kwargs
        ):
        '''AnyNet keras implementation

            Args:
                learning_rate (float): Adam optimizer
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
        super(AnyNetV2, self).__init__()

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
        self.batch_size = None
        self._disparity_network = []
        self._batch_size_observers = []
        # if with_cspn:
        #     self.cspnet = ConvSpatialPropagationNet(
        #         cspn_conv2d_filters,
        #         cspn_conv2d_step
        #     )
    def get_config(self):
        config = {
            'unet_conv2d_filters': self.unet_conv2d_filters,
            'unet_nblocks': self.unet_nblocks,
            'disp_conv3d_filters': self.disp_conv3d_filters,
            'disp_conv3d_layers': self.disp_conv3d_layers,
            'disp_conv3d_growth_rate': self.disp_conv3d_growth_rate,
            'cspn_conv3d_filters': self.cspn_conv3d_filters,
            'with_cspn': self.with_cspn,
            'local_max_disps': self.local_max_disps,
            'global_max_disp': self.global_max_disp,
            'stages': self.stages, 
        }
        config.update(super().get_config())
        return config

    def build(self, input_shape):
        
        left_img = keras.Input(shape=input_shape[0][1:], name="anynet_left_img")
        right_img = keras.Input(shape=input_shape[1][1:], name="anynet_right_img")

        _, height, width, _ = input_shape[0]

        self.feature_extractor  = FeatureExtractor(
            init_filters=self.unet_conv2d_filters,
            nblocks=self.unet_nblocks,
        )
        feats_l = self.feature_extractor(left_img)
        feats_r = self.feature_extractor(right_img)

        outputs = []
        for stage in range(self.stages):
            if stage == 0:
                disparity_stage = DisparityNetworkStage0(
                    disp_conv3d_filters=self.disp_conv3d_filters,
                    disp_conv3d_layers=self.disp_conv3d_layers,
                    disp_conv3d_growth_rate=self.disp_conv3d_growth_rate[stage],
                    local_max_disp=self.local_max_disps[stage],
                    height=height,
                    width=width
                )

                self._disparity_network.append(disparity_stage)
                disparity_stage.batch_size = self.batch_size
                disparity_stage = disparity_stage([feats_l[stage], feats_r[stage]])
                outputs.append(disparity_stage)

            else:
                disparity_stage = DisparityNetworkStageN(
                    disp_conv3d_filters=self.disp_conv3d_filters,
                    disp_conv3d_layers=self.disp_conv3d_layers,
                    disp_conv3d_growth_rate=self.disp_conv3d_growth_rate[stage],
                    local_max_disp=self.local_max_disps[stage],
                    stage=stage,
                    height=height,
                    width=width,
                    name=f'disparity_network_stage{stage}'
                )

                self._disparity_network.append(disparity_stage)
                disparity_stage.batch_size = self.batch_size
                disparity_stage = disparity_stage([
                    outputs[stage-1],
                    [feats_l[stage], feats_r[stage]]
                ])
                outputs.append(disparity_stage)

        self.outputs = outputs
        self.model = keras.Model([left_img, right_img], {
            f'disparity-{i}': x for i,x in enumerate(outputs)
        }, name="anynet")
        return self.model
    
    def call(self, inputs):
        return self.model(inputs)

class AnyNetFactory(object):

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
        super(AnyNetFactory, self).__init__()

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

    def build(self, input_shape):

        self.anynet = AnyNetV2(
                unet_conv2d_filters=self.unet_conv2d_filters,
                unet_nblocks=self.unet_nblocks,
                cspn_conv3d_filters=self.cspn_conv3d_filters,
                local_max_disps=self.local_max_disps,
                global_max_disp=self.global_max_disp,
                stages=self.stages,
        )
        self.anynet.batch_size = input_shape[0][0]
        return self.anynet.build(input_shape)
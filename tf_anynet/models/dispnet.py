import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.image import dense_image_warp

from .regularization import Conv3DRegularizer


@keras.utils.register_keras_serializable(package='AnyNet')
class CostVolume2DV2(keras.layers.Layer):
    def __init__(self, local_max_disp, stride=1):
        assert local_max_disp % stride == 0
        super(CostVolume2DV2, self).__init__()
        self.local_max_disp = local_max_disp
        self.stride = stride
        self.height = None
        self.width = None

    def get_config(self):
        config = {
            'stride': self.stride,
            'local_max_disp': self.local_max_disp,
            'height': self.height,
            'width': self.width
        }
        config.update(super().get_config())
        return config
    def build(self, input_shape):
        input_l, _= input_shape
        _, height, width, _ = input_l
        self.height = height
        self.width = width

    def call(self, inputs):
        feat_l, feat_r = inputs

        cost = []
        for k in range(0, self.local_max_disp, self.stride):

            if k > 0:
                k_init = tf.abs(feat_l[:, :, :k, :])
                k_dim = tf.abs(feat_l[:, :, k:, :] - feat_r[:, :, :-k, :])
                k_dim = tf.concat([k_init, k_dim], -2)
                norm = tf.linalg.norm(
                    k_dim, 
                    ord=1, 
                    axis=-1,
                )
                cost.append(norm)
            else:
                # no k slice initializer needed in the top-left corner
                k_dim = tf.abs(feat_l[:, :, :, :] - feat_r[:, :, :, :])
                norm = tf.linalg.norm(
                    k_dim, 
                    ord=1, 
                    axis=-1,
                )
                cost.append(norm)

        return tf.stack(cost, axis=-1)

@keras.utils.register_keras_serializable(package='AnyNet')
class CostVolume2D(keras.layers.Layer):
    def __init__(self, local_max_disp, stride=1, batch_size=None):
        assert local_max_disp % stride == 0
        super(CostVolume2D, self).__init__()
        self.local_max_disp = local_max_disp
        self.stride = stride
        self.batch_size = batch_size
        self.height = None
        self.width = None

    def get_config(self):
        config = {
            'stride': self.stride,
            'local_max_disp': self.local_max_disp,
            'batch_size': self.batch_size
        }
        config.update(super().get_config())
        return config

    def build(self, input_shape):
        input_l, _= input_shape
        _, height, width, _ = input_l
        self.height = height
        self.width = width
        self.cost = tf.Variable(
            initial_value=self.initialize_cost2d(),
            #validate_shape=False, 
            trainable=False, 
            shape=tf.TensorShape((None, self.height, self.width, self.local_max_disp//self.stride)),
            dtype=tf.float32
            )

    def initialize_cost2d(self):
        return tf.zeros((self.batch_size, self.height, self.width, self.local_max_disp//self.stride))
    
    def call(self, inputs):
        feat_l, feat_r = inputs
        

        cost = self.cost.assign(self.initialize_cost2d())
        
        for i in range(0, self.local_max_disp, self.stride):
            reduced = tf.reduce_sum(
                tf.math.abs(feat_l[:, :, :i, :]),
                axis=-1
            )
            cost = cost[:, :, :i, i//self.stride].assign(reduced)
            if i > 0:
                _, norm = tf.linalg.normalize(
                    feat_l[:, :, i:, :] - feat_r[:, :, :-i, :], 
                    ord=1, 
                    axis=-1
                )
                cost = cost[:, :, i:, i//self.stride].assign(
                    tf.squeeze(norm, axis=-1)
                )
            else:
                _, norm = tf.linalg.normalize(
                    feat_l[:, :, :, :] - feat_r[:, :, :, :], 
                    ord=1, 
                    axis=-1
                )
                cost[:, :, i:, i//self.stride].assign(
                    tf.squeeze(norm, axis=-1)
                )

        return cost

@keras.utils.register_keras_serializable(package='AnyNet')
class CostVolume3D(keras.layers.Layer):
    def __init__(self, local_max_disp, batch_size=None, stride=1):
        super(CostVolume3D, self).__init__()
        self.local_max_disp = local_max_disp
        self.stride = stride
        self.batch_size = batch_size

    def get_config(self):
        config = {
            'stride': self.stride,
            'local_max_disp': self.local_max_disp,
            'batch_size': self.batch_size
        }
        config.update(super().get_config())
        return config

    def warp(self, x, disp):
        #import pdb; pdb.set_trace()
        return dense_image_warp(x, -disp)

    def call(self, inputs):
        feat_l, feat_r, wflow = inputs

        _, height, width, channels = feat_r.shape

        batch_disp = tf.tile(
            wflow[:,:,:,:,None],
            multiples=(1, 1, 1, 1, self.local_max_disp*2-1)
        )

        batch_disp = tf.reshape(batch_disp, (-1, height, width, 1))

        batch_shift = tf.tile(
            range(-self.local_max_disp+1, self.local_max_disp),
            multiples=(self.batch_size,)
        )[:,None,None,None] * self.stride

        batch_shift = tf.cast(batch_shift, tf.float32)


        batch_disp = batch_disp - batch_shift

        batch_feat_l = tf.tile(
            feat_l[:,:,:,:, None],
            multiples=(1, 1, 1, 1, self.local_max_disp*2-1)
        )

        batch_feat_l = tf.reshape(batch_feat_l,
            (-1, height, width, channels)
        )

        batch_feat_r = tf.tile(
            feat_r[:,:,:,:, None],
            multiples=(1, 1, 1, 1, self.local_max_disp*2-1)
        )
        batch_feat_r = tf.reshape(batch_feat_r,
            (-1, height, width, channels)
        )
        
        batch_r_warped = self.warp(batch_feat_r, batch_disp)
        batch_l_diff = batch_feat_l - batch_r_warped

        norm = tf.linalg.norm(batch_l_diff, ord=1, axis=-1, keepdims=True)

        cost = tf.reshape(norm, (self.batch_size,height,width, -1))

        return cost

@keras.utils.register_keras_serializable(package='AnyNet')
class DisparityNetworkStage0(keras.layers.Layer):
    def __init__(self, 
        disp_conv3d_filters=4,
        disp_conv3d_layers=4,
        disp_conv3d_growth_rate=1,
        local_max_disp=3,
        height=None,
        width=None,
        stride=1,
        batch_size=None,
        *args, **kwargs
        ):
        super(DisparityNetworkStage0, self).__init__()

        self.disp_conv3d_filters = disp_conv3d_filters
        self.disp_conv3d_layers = disp_conv3d_layers
        self.disp_conv3d_growth_rate = disp_conv3d_growth_rate
        self.local_max_disp = local_max_disp
        self.stage = 0
        self.height = height
        self.width = width
        self.stride = stride
        self.batch_size = batch_size
        self.cost_volume = CostVolume2DV2(self.local_max_disp, stride=self.stride)

        self.regularizer = Conv3DRegularizer(
            self.disp_conv3d_layers,
            self.disp_conv3d_filters*self.disp_conv3d_growth_rate
        )
        
    def get_config(self):
        config = {
            'disp_conv3d_filters': self.disp_conv3d_filters,
            'disp_conv3d_layers': self.disp_conv3d_layers,
            'disp_conv3d_growth_rate': self.disp_conv3d_growth_rate,
            'local_max_disp': self.local_max_disp,
            'stage': self.stage,
            'width': self.width,
            'height': self.height,
            'stride': self.stride,
            'batch_size': self.batch_size
        }
        config.update(super().get_config())
        return config
    
    def call(self, inputs):

        feat_l, feat_r = inputs

        cost = self.cost_volume([feat_l, feat_r])

        # expand dimensions to preserve "context" in 3D convolution
        cost = tf.expand_dims(cost, -1)
        cost = self.regularizer(cost)

        cost = tf.squeeze(cost, axis=-1)
        
        softmax = keras.layers.Softmax(axis=-1, name=f'softmax{self.stage}')(-cost)
        pred_low_res = DisparityRegression(
            0, self.local_max_disp
        )(softmax)
        pred_low_res = pred_low_res * self.height * self.width
        output = tf.image.resize(pred_low_res, [self.height, self.width])
        return output


@keras.utils.register_keras_serializable(package='AnyNet')
class DisparityNetworkStageN(keras.layers.Layer):
    def __init__(self, 
        disp_conv3d_filters=4,
        disp_conv3d_layers=4,
        disp_conv3d_growth_rate=1,
        local_max_disp=3,
        height=None,
        width=None,
        stride=1,
        stage=None,
        batch_size=None,
        name=None,
        *args, **kwargs
        ):
        super(DisparityNetworkStageN, self).__init__(name=name)

        self.disp_conv3d_filters = disp_conv3d_filters
        self.disp_conv3d_layers = disp_conv3d_layers
        self.disp_conv3d_growth_rate = disp_conv3d_growth_rate
        self.local_max_disp = local_max_disp
        self.stage = stage
        self.height = height
        self.width = width
        self.stride = stride
        self.batch_size = batch_size
        self.cost_volume = CostVolume3D(self.local_max_disp, stride=self.stride)
        self.regularizer = Conv3DRegularizer(
            self.disp_conv3d_layers,
            self.disp_conv3d_filters*self.disp_conv3d_growth_rate
        )
        
    def get_config(self):
        config = {
            'disp_conv3d_filters': self.disp_conv3d_filters,
            'disp_conv3d_layers': self.disp_conv3d_layers,
            'disp_conv3d_growth_rate': self.disp_conv3d_growth_rate,
            'local_max_disp': self.local_max_disp,
            'stage': self.stage,
            'width': self.width,
            'height': self.height,
            'stride': self.stride,
            'batch_size': self.batch_size,
        }
        config.update(super().get_config())
        return config
    
    def call(self, inputs):
        residuals, feats = inputs
        feats_l, feats_r = feats

        self.cost_volume.batch_size = self.batch_size
        
        wflow = tf.image.resize(residuals, (feats_l.shape[1], feats_l.shape[2]))
        wflow = wflow * feats_l.shape[1] / self.height

        cost = self.cost_volume(
            [feats_l, feats_r, wflow]
        )
        cost = tf.expand_dims(cost, -1)
        cost = self.regularizer(cost)
        cost = tf.squeeze(cost, axis=-1)
        
        softmax = keras.layers.Softmax(axis=-1, name=f'softmax{self.stage}')(-cost)
        start = -self.local_max_disp+1
        end = self.local_max_disp
        pred_low_res = DisparityRegression(start, end)(softmax)
        pred_low_res = pred_low_res * self.height * self.width
        resized = tf.image.resize(pred_low_res, [ self.height, self.width ])
        return resized + residuals
    
 
@keras.utils.register_keras_serializable(package='AnyNet')
class DisparityRegression(keras.layers.Layer):
    def __init__(self, start, end, stride=1, *args, **kwargs):
        super(DisparityRegression, self).__init__()
        self.start = start

        self.end = end
        self.stride = stride

    def get_config(self):
        config = {
            'start': self.start,
            'end': self.end
        }
        config.update(super().get_config)
        return config

    def build(self, input_shape):
        self.disp = tf.reshape(
            range(self.start*self.stride, self.end*self.stride, self.stride),
            (1, 1, 1, -1)
        )
        # multiples = tf.constant((input_shape[1],input_shape[2],1))
        # self.disp = tf.tile(
        #     self.disp,
        #     multiples=multiples,
        # )

        self.disp = tf.cast(self.disp, tf.float32)
        

    def call(self, inputs):
        #x = x * self.disp

        disp = tf.vectorized_map(
            lambda x: x * self.disp, inputs
        )
        disp = tf.reshape(disp, (
            -1,
            disp.shape[-3],
            disp.shape[-2],
            disp.shape[-1] * disp.shape[1]
        ))
        return keras.backend.sum(
            disp, axis=-1, keepdims=True
        )

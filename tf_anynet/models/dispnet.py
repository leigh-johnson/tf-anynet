import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.image import dense_image_warp

from .regularization import Conv3DRegularizer

@keras.utils.register_keras_serializable(package='AnyNet')
class DisparityNetwork(keras.layers.Layer):

    def __init__(self, 
        disp_conv3d_filters=4,
        disp_conv3d_layers=4,
        disp_conv3d_growth_rate=[4, 1, 1],
        local_max_disps=[12, 3, 3],
        global_max_disp=192,
        stages=3,
        batch_size=8,
        *args, **kwargs
        ):
        super(DisparityNetwork, self).__init__()

        self.disp_conv3d_filters = disp_conv3d_filters
        self.disp_conv3d_layers = disp_conv3d_layers
        self.disp_conv3d_growth_rate = disp_conv3d_growth_rate
        self.local_max_disps = local_max_disps
        self.global_max_disp = global_max_disp
        self.stages = stages
        self.batch_size = batch_size
        self.width = None
        self.height = None
        self.cost2d = None

        self.volume_postprocess = []
        self.bnorms = [keras.layers.BatchNormalization() for n in range(0,stages)]

        for i in range(len(disp_conv3d_growth_rate)):
            regularizer = Conv3DRegularizer(
                self.disp_conv3d_layers,
                self.disp_conv3d_filters*self.disp_conv3d_growth_rate[i]
            ) 
            self.volume_postprocess.append(regularizer)     
    
    def get_config(self):
        config = {
            'disp_conv3d_filters': self.disp_conv3d_filters,
            'disp_conv3d_layers': self.disp_conv3d_layers,
            'disp_conv3d_growth_rate': self.disp_conv3d_growth_rate,
            'local_max_disps': self.local_max_disps,
            'global_max_disp': self.global_max_disp,
            'stages': self.stages,
            'batch_size': self.batch_size
        }
        config.update(super().get_config())
        return config

    def warp(self, x, disp):
        return dense_image_warp(x, -disp)

    def warpV2(self, x, disp):
        batch_size, height, width, channels = (
            _get_dim(x, 0),
            _get_dim(x, 1),
            _get_dim(x, 2),
            _get_dim(x, 3),
        )

        xx = tf.range(0, width)
        xx = tf.reshape(xx, (1, -1))
        xx = tf.tile(xx,multiples=(height, 1))
        xx = tf.reshape(xx, (1,height, width, 1))
        xx = tf.tile(xx, multiples=(batch_size,1,1,1))
        xx = tf.cast(xx, tf.float32)

        yy = tf.range(0, height)
        yy = tf.reshape(yy, (1, -1))
        yy = tf.tile(yy, multiples=(1, width))
        yy = tf.reshape(yy, (1, height, width, 1))
        yy = tf.tile(yy, multiples=(batch_size,1,1,1))
        yy = tf.cast(yy, tf.float32)

        vgrid = tf.concat([xx,yy], axis=-1)

        vgrid1 = vgrid[:,:,:,:1] - disp
        vgrid = tf.concat([vgrid1, vgrid[:,:,:,1:]], axis=-1)

        import pdb; pdb.set_trace()
        return vgrid
    
    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride

        if self.cost2d is None:
            cost = tf.Variable(
                lambda : tf.zeros((self.batch_size, feat_l.shape[1], feat_l.shape[2], maxdisp//stride))
            )
            self.cost2d = cost
        else:
            cost = self.cost2d

        for i in range(0, maxdisp, stride):
            reduced = tf.reduce_sum(
                tf.math.abs(feat_l[:, :, :i, :]),
                axis=-1
            )
            cost[:, :, :i, i//stride].assign(reduced)
            # @todo graph norms or publish as metrics
            if i > 0:
                _, norm = tf.linalg.normalize(
                    feat_l[:, :, i:, :] - feat_r[:, :, :-i, :], 
                    ord=1, 
                    axis=-1
                )
                cost[:, :, i:, i//stride].assign(
                    tf.squeeze(norm, axis=-1)
                )
            else:
                _, norm = tf.linalg.normalize(
                    feat_l[:, :, :, :] - feat_r[:, :, :, :], 
                    ord=1, 
                    axis=-1
                )
                cost[:, :, i:, i//stride].assign(
                    tf.squeeze(norm, axis=-1)
                )

        return cost

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        
        size = feat_l.shape
        
        batch_disp = tf.tile(
            disp[:,:,:,:,None],
            multiples=(1, 1, 1, 1, maxdisp*2-1)
        )

        batch_disp = tf.reshape(batch_disp, (-1, size[1], size[2], 1))

        batch_shift = tf.tile(
            range(-maxdisp+1, maxdisp),
            multiples=(self.batch_size,)
        )[:,None,None,None] * stride

        batch_shift = tf.cast(batch_shift, tf.float32)


        batch_disp = batch_disp - batch_shift

        batch_feat_l = tf.tile(
            feat_l[:,:,:,:, None],
            multiples=(1, 1, 1, 1, maxdisp*2-1)
        )

        batch_feat_l = tf.reshape(batch_feat_l,
            (-1, size[1], size[2], size[3])
        )

        batch_feat_r = tf.tile(
            feat_r[:,:,:,:, None],
            multiples=(1, 1, 1, 1, maxdisp*2-1)
        )
        batch_feat_r = tf.reshape(batch_feat_r,
            (-1, size[1], size[2], size[3])
        )

        batch_r_warped = self.warp(batch_feat_r, batch_disp)
        batch_l_diff = batch_feat_l - batch_r_warped

        norm = tf.linalg.norm(batch_l_diff, ord=1, axis=-1, keepdims=True)

        cost = tf.reshape(norm, (self.batch_size,size[1],size[2], -1))

        return cost

    def call(self, inputs):

        feats_l, feats_r = inputs
        
        pred = []
        for scale in range(len(feats_l)):
            if scale > 0:
                wflow = tf.image.resize(
                    pred[scale-1], 
                    (feats_l[scale].shape[1], feats_l[scale].shape[2]
                ))
                wflow = wflow * feats_l[scale].shape[1] / self.height


                cost = self._build_volume_2d3(
                    feats_l[scale], feats_r[scale],
                    self.local_max_disps[scale], wflow)

                cost = tf.expand_dims(cost, -1)
                cost = self.volume_postprocess[scale](cost)
                cost = tf.squeeze(cost, axis=-1)
                
                softmax = layers.Softmax(axis=-1, name=f'softmax{scale}')(-cost)

                start = -self.local_max_disps[scale]+1
                end = self.local_max_disps[scale]
                pred_low_res = DisparityRegression(start, end)(softmax)
                #pred_low_res = self.bnorms[scale](pred_low_res)
                disp_up = tf.image.resize(pred_low_res, [ self.height, self.width ])
                pred.append(disp_up+pred[scale-1])
            else:
                cost = self._build_volume_2d(feats_l[scale], feats_r[scale],
                                             self.local_max_disps[scale])

                cost = tf.expand_dims(cost, -1)
                cost = self.volume_postprocess[scale](cost)
                cost = tf.squeeze(cost)
                softmax = layers.Softmax(axis=-1, name=f'softmax{scale}')(-cost)     
                pred_low_res = DisparityRegression(
                    0, self.local_max_disps[scale]
                )(softmax)
                #pred_low_res = self.bnorms[scale](pred_low_res)
                disp_up = tf.image.resize(pred_low_res, [self.height, self.width])
                pred.append(disp_up)

        #pred = [self.bnorms[i](x) for i,x in enumerate(pred)]
        #pred = [tf.image.per_image_standardization(x) for x in pred]
        return pred

@keras.utils.register_keras_serializable(package='AnyNet')
class DisparityRegression(keras.layers.Layer):
    def __init__(self, start, end, stride=1):
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

        multiples = tf.constant((input_shape[0], input_shape[1],input_shape[2],1))
        self.disp = tf.tile(
            self.disp,
            multiples=multiples,
        )

        self.disp = tf.cast(self.disp, tf.float32)
        

    def call(self, x):
        x = x * self.disp
        return keras.backend.sum(
            x, axis=-1, keepdims=True
        )
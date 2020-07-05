
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.image import dense_image_warp

from .regularization import Conv3DRegularizer
from .feature_extractor import FeatureExtractor

class L1DisparityMaskLoss(object):

    __name__ = 'L1DisparityMaskLoss'
    def __init__(self, weights, stages, global_max_disp):
        self.weights = weights
        self.stages = stages
        self.global_max_disp = global_max_disp
        self.loss = keras.losses.Huber()

    def __call__(self, disp, logits):
        mask = disp[0] < self.global_max_disp
        # if sum(mask) == 0:
        #     raise Exception('disparty mask is empty and i have no idea why (: (should never happen)')
        
        # stop gradient backpropagation on mask
        #import pdb; pdb.set_trace()
        mask = tf.stop_gradient(mask)

        loss = [
            self.weights[x] * 
            self.loss(
                logits[x][mask], disp[x][mask])
            for x in range(self.stages)
        ]

        return sum(loss)


class DisparityRegression(keras.layers.Layer):
    def __init__(self, start, end, stride=1):
        super(DisparityRegression, self).__init__()

        self.disp = tf.reshape(
            tf.range(start*stride, end*stride, stride),
            (1, 1, 1, -1)
        )

    def call(self, x):
        disp = tf.cast(
            tf.tile(
            self.disp,
            multiples=(x.shape[0], x.shape[1],x.shape[2],1),
            ),
            dtype=tf.float32
        )
        return keras.backend.sum(
            x * disp, axis=-1, keepdims=True
        )

class AnyNet(keras.Model):

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
        input_dim=(256, 512), # height, width
        stages=3,
        batch_size=8,
        eval_samples=None,
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
        super(AnyNet, self).__init__()

    
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
        self.input_dim = input_dim
        self.stages = stages
        self.eval_samples = eval_samples# .map(lambda imgL,imgR,disp: (imgL, imgR))
        #self.eval_gt = eval_samples.map(lambda imgL,imgR,disp: disp)
        #self.batch_size = 

        self.feature_extractor  = FeatureExtractor(
            init_filters=unet_conv2d_filters,
            nblocks=unet_nblocks,
            input_dim=input_dim,
            batch_size=batch_size
        )

        self.volume_postprocess = []

        for i in range(len(disp_conv3d_growth_rate)):
            regularizer = Conv3DRegularizer(
                self.disp_conv3d_layers,
                self.disp_conv3d_filters*self.disp_conv3d_growth_rate[i]
            ) 
            # net3d = conv3d_net(disp_conv3d_layers, disp_conv3d_filters*disp_conv3d_growth_rate[i])
            self.volume_postprocess.append(regularizer)       

    def test_step(self, data):
        imgL, imgR, dispL = data

        # Compute predictions
        disp = tf.tile(tf.expand_dims(dispL, axis=0), multiples=(3,1,1,1,1))
        
        logits = self([imgL, imgR], training=False)
        # Updates the metrics tracking the loss

        self.compiled_loss(disp, logits)
        # Update the metrics.
        self.compiled_metrics.update_state(disp, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):

        imgL, imgR, dispL = data
        #write_step_imgs(data)

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        
        # tile dispL to match shape of logits
        disp = tf.tile(tf.expand_dims(dispL, axis=0), multiples=(3,1,1,1,1))

        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.

            logits = self([imgL, imgR], training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            # loss_value = self.compiled_loss(logits, dispL, regularization_losses=self.losses)
            # @todo calculate regularization losses
            loss = self.compiled_loss(
                disp, 
                logits
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(disp, logits)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def warp(self, x, disp):
        return dense_image_warp(x, disp)

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        
        size = feat_l.shape
        
        batch_disp = tf.tile(
            disp[:,:,:,:,None],
            multiples=(1, 1, 1, 1, maxdisp*2-1)
        )

        batch_disp = tf.reshape(batch_disp, (-1, size[1], size[2], 1))
        
        batch_shift = tf.tile(
            tf.range(-maxdisp+1, maxdisp),
            multiples=(size[0],)
        )[:,None,None,None] * stride

        batch_shift = tf.cast(batch_shift, tf.float32)


        batch_disp = batch_disp - batch_shift

        batch_feat_l = tf.tile(
            feat_l[:,:,:,:,None],
            multiples=(1, 1, 1, 1, maxdisp*2-1)
        )
        batch_feat_l = tf.reshape(batch_feat_l,
            (-1, size[1], size[2], size[3])
        )

        batch_feat_r = tf.tile(
            feat_r[:,:,:,:,None],
            multiples=(1, 1, 1, 1, maxdisp*2-1)
        )
        batch_feat_r = tf.reshape(batch_feat_r,
            (-1, size[1], size[2], size[3])
        )

        norm = tf.linalg.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), ord=1, axis=-1)

        cost = tf.reshape(norm, (size[0],size[1],size[2], -1))
        return cost

    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride

        cost = tf.Variable(
            tf.zeros((feat_l.shape[0], feat_l.shape[1], feat_l.shape[2], maxdisp//stride))
        )
        

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
        
    def call(self, imgs):
        
        left_img, right_img = imgs


        img_size = left_img.shape
        
        feats_l = self.feature_extractor(left_img)
        feats_r = self.feature_extractor(right_img)

        pred = []

        for scale in range(len(feats_l)):
            if scale > 0:
                wflow = tf.image.resize(
                    pred[scale-1], 
                    (feats_l[scale].shape[1], feats_l[scale].shape[2]
                )) * feats_l[scale].shape[1] / img_size[1]

                cost = self._build_volume_2d3(
                    feats_l[scale], feats_r[scale],
                    self.local_max_disps[scale], wflow)
            else:
                cost = self._build_volume_2d(feats_l[scale], feats_r[scale],
                                             self.local_max_disps[scale])
            cost = tf.expand_dims(cost, -1)
            cost = self.volume_postprocess[scale](cost)
            cost = tf.squeeze(cost)
            
            softmax = layers.Softmax(axis=-1)(-cost)

            if scale == 0:
                pred_low_res = DisparityRegression(0, self.local_max_disps[scale])(softmax)

                disp_up = tf.image.resize(pred_low_res, img_size[1:3])
                pred.append(disp_up)
            else:

                pred_low_res = DisparityRegression(-self.local_max_disps[scale]+1, self.local_max_disps[scale])(softmax)
                disp_up = tf.image.resize(pred_low_res, img_size[1:3])
                pred.append(disp_up+pred[scale-1])
        
        # @todo calculate regularization losses
        self.depth_map = tf.convert_to_tensor(pred, dtype=tf.float32)
        return self.depth_map
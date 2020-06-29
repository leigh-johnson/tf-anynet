
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .feature_extractor_fn import FeatureExtractor, conv3d_net  

class L1DisparityMaskLoss(object):

    __name__ = 'L1DisparityMaskLoss'
    def __init__(self, weights, stages, global_max_disp):
        self.weights = weights
        self.stages = stages
        self.global_max_disp = global_max_disp

    def __call__(self, disp, logits):
        mask = disp < self.global_max_disp
        import pdb; pdb.set_trace()
        if tf.math.reduce_sum(mask) == 0:
            raise Exception('disparty mask is empty and i have no idea why (: (should never happen)')
        
        # stop gradient backpropagation on mask
        mask = tf.stop_gradient(mask)
        
        loss = [
            self.weights[x] * keras.losses.huber(outputs[x][mask], disp[mask])
            for x in range(self.stages)
        ]
        return sum(loss)


class DisparityRegression(keras.layers.Layer):
    def __init__(self, start, end, stride=1):
        super(DisparityRegression, self).__init__()

        self.disp = tf.reshape(
            tf.range(start*stride, end*stride, stride),
            (1, -1, 1, 1)
        )

    def call(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        return keras.backend.sum(
            x * disp, axis=1, keepdims=True
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

        
        # right_img = keras.Input(shape=(3, input_dim[0], input_dim[1]), name="right_img_input")
        # self.inputs = (left_img, right_img)
        
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
        #self.batch_size = 

        self.feature_extractor  = FeatureExtractor(
            init_filters=unet_conv2d_filters,
            nblocks=unet_nblocks,
            input_dim=input_dim,
            batch_size=batch_size
        )

        # self.volume_postprocess = []

        # for i in range(3):
        #     net3d = conv3d_net(disp_conv3d_layers, disp_conv3d_filters*disp_conv3d_growth_rate[i])
        #     self.volume_postprocess.append(net3d)       

    def test_step(self, data):
        (imgL, imgR), dispL = data

        # Compute predictions
        logits = self(imgL, imgR, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(dispL, logits)
        # Update the metrics.
        self.compiled_metrics.update_state(dispL, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def write_step_imgs(self, data):
        (imgL, imgR), dispL = data
        imgL = tf.image.encode_png(
            imgL, compression=-1
        )
        tf.io.write_file(
            '', contents, name=None
        )
        imgR = tf.image.encode_png(
            imgR, compression=-1
        )

        dispL = tf.image.encode_png(
            dispL, compression=-1
        )


    def train_step(self, data):
        (imgL, imgR), dispL = data
        #write_step_imgs(data)

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = self([imgL, imgR], training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            # loss_value = self.compiled_loss(logits, dispL, regularization_losses=self.losses)
            # @todo calculate regularization losses
            import pdb; pdb.set_trace()
            loss = self.compiled_loss(
                dispL, 
                logits
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(dispL, logits)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def warp(self, x, disp):
        '''
            Warp img2 back to img1
        x: [B, H, W, C] (img2)
        optical  flow: [B, 2, H, W] 
        '''

        B, H, W, C = x.shape()

        xx = tf.range(0, W).reshape(1, -1).repeat(H, 1)
        yy = tf.range(0, H).reshape(-1, 1).repeat(1, W)
        xx = xx.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = tf.concat((xx, yy), 1)

        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output
    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        size = feat_l.shape()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).reshape(-1,1,size[-2], size[-1])
        batch_shift = tf.range(-maxdisp+1, maxdisp).repeat(size[0])[:,None,None,None] * stride
        batch_disp = batch_disp - batch_shift
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).reshape(-1,size[-3],size[-2], size[-1])
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).reshape(-1,size[-3],size[-2], size[-1])
        
        cost = tf.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), ord=1, axis=-1)
        cost = cost.reshape(size[0],-1, size[2],size[3])
        return cost

    def call(self, imgs):
        left_img, right_img = imgs
        #import pdb; pdb.set_trace()

        #img_size = left_img.shape()
        
        feats_l = self.feature_extractor(left_img)
        feats_r = self.feature_extractor(right_img)
        return [feats_l, feats_r]
        #pred = []

        # for scale in range(len(feats_l)):
        #     if scale > 0:
        #         wflow = layers.UpSampling3D(
        #             pred[scale-1],
        #             size=(feats_l[scale].size(1), feats_l[scale].size(2)),
        #             interpolation='bilinear',
        #             ) * feats_l[scale].size(1) / img_size[2]

        #         cost = self._build_volume_2d3(feats_l[scale], feats_r[scale],
        #                                  self.local_max_disps[scale], wflow, stride=1)
        #     else:
        #         cost = self._build_volume_2d(feats_l[scale], feats_r[scale],
        #                                      self.local_max_disps[scale], stride=1)
        #     cost = tf.expand_dims(cost, -1)
        #     cost = self.volume_postprocess[scale](cost)
        #     cost = cost.squeeze(-1)
            
        #     softmax = layers.Softmax(axis=-1)(-cost)

        #     if scale == 0:
        #         pred_low_res = DisparityRegression(0, self.local_max_disps[scale])(softmax)
        #         pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(1)
        #         disp_up = layers.UpSampling3D(
        #             pred_low_res,
        #             size=(img_size[2], img_size[3]),
        #             mode='bilinear'
        #         )
        #         pred.append(disp_up)
        #     else:
        #         pred_low_res = DisparityRegression(self.local_max_disps[scale]+1, self.local_max_disps[scale], stride=1)(softmax)
        #         pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(1)
        #         disp_up = layers.UpSampling3D(
        #             pred_low_res,
        #             size=(img_size[2], img_size[3]),
        #             mode='bilinear'
        #         )
        #         pred.append(disp_up+pred[scale-1])
        
        # # @todo calculate regularization losses
        # return pred

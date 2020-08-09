
import tensorflow as tf
from tensorflow import keras


def masked_pixel_ratio(disp, global_max_disp):
    excluded = tf.reduce_sum(tf.cast(disp > global_max_disp, tf.float32))
    included = tf.reduce_sum(tf.cast(disp <= global_max_disp, tf.float32))
    total = excluded + included
    return included/total

@keras.utils.register_keras_serializable(package='AnyNet')
class L1DisparityMaskLoss(keras.losses.Loss):

    __name__ = 'L1DisparityMaskLoss'
    def __init__(self, stage, global_max_disp, *args, **kwargs):
        super(L1DisparityMaskLoss, self).__init__()
        self.stage = stage
        self.global_max_disp = global_max_disp
        self.loss = keras.losses.Huber(
            reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )

    def get_config(self):
        config = {
            'stage': self.stage,
            'global_max_disp': self.global_max_disp
        }
        config.update(super().get_config())
        return config
    def __call__(self, disp, logits, sample_weight=None):

        #disp = tf.squeeze(disp, -1)

        mask = disp <= self.global_max_disp

        #squeezed = tf.squeeze(logits, -1)
        
        loss = self.loss(logits[mask], disp[mask])

        if sample_weight:
            loss * sample_weight
            
        return loss

import tensorflow as tf
from tensorflow import keras

class L1DisparityMaskLoss(keras.losses.Loss):

    __name__ = 'L1DisparityMaskLoss'
    def __init__(self, stage, global_max_disp):
        super(L1DisparityMaskLoss, self).__init__()
        self.stage = stage
        self.global_max_disp = global_max_disp
        self.loss = keras.losses.Huber()

    def __call__(self, disp, logits, sample_weight=None):
        
        mask = disp < self.global_max_disp
        mask = tf.stop_gradient(mask)

        loss = self.loss(logits[mask], disp[mask])

        if sample_weight:
            return loss * sample_weight
        return loss

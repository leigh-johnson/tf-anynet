
import tensorflow as tf
from tensorflow import keras

class L1DisparityMaskLoss(object):

    __name__ = 'L1DisparityMaskLoss'
    def __init__(self, weights, stages, global_max_disp):
        self.weights = weights
        self.stages = stages
        self.global_max_disp = global_max_disp
        self.loss = keras.losses.Huber()

    def __call__(self, disp, logits):

        mask = disp < self.global_max_disp
        mask = tf.stop_gradient(mask)

        logits = tf.expand_dims(logits, axis=-1)
        loss = [
            self.weights[x] * 
            self.loss(
                logits[x][mask], disp[mask])
            for x in range(self.stages)
        ]

        return sum(loss)

@keras.utils.register_keras_serializable(package='AnyNet')
class RootMeanSquaredError(keras.metrics.RootMeanSquaredError):

    def __init__(self):
        super(RootMeanSquaredError, self).__init__()


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.tile(tf.expand_dims(y_true, axis=0), multiples=(3,1,1,1,1))
        return super(RootMeanSquaredError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

@keras.utils.register_keras_serializable(package='AnyNet')
class MeanAbsolutePercentageError(keras.metrics.MeanAbsolutePercentageError):

    def __init__(self):
        super(MeanAbsolutePercentageError, self).__init__()


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.tile(tf.expand_dims(y_true, axis=0), multiples=(3,1,1,1,1))
        return super(MeanAbsolutePercentageError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    # def get_config(self):
    # config = {
    #     'init_filters': self.init_filters,
    #     'nblocks': self.nblocks,
    #     'batch_size': self.batch_size
    # }
    # config.update(super().get_config())
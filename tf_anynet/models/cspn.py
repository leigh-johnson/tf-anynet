import tensorflow as tf
from tensorflow import keras


class ConvSpatialPropagationNet(keras.layers.Layer):
    
    def __init__(self, in_channels=8, step=24, kernel=3):

        self.in_channels = in_channels
        self.out_channels = 1
        self.step = step
        self.kernel = kernel

    
    def build(self, input_shape):
        self.sum_conv = keras.layers.Conv3D(
                in_channels=self.in_channels,
                out_channels=1,
                kernel_size=(1, 1, 1),
                stride=1,
                padding=0,
                bias=False
            )
    
    def call(self, guidance, blur_depth, sparse_depth):

        gate_wb, gate_sum = self.affinity_normalization(guidance)
        raw_depth_input = blur_depth
        result_depth = blur_depth

        sparse_mask = sparse_depth.sign()

        for i in range(self.step):
            # one propagation
            neigbor_weighted_sum = self.sum_conv(gate_wb * result_depth)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, 1:-1, 1:-1, :]
            result_depth = neigbor_weighted_sum

            result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth

            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input

        return result_depth

    def affinity_normalization(self, guidance):

        gate1_wb_cmb = tf.slice(guidance[-1], 0, self.out_channels)
        gate2_wb_cmb = tf.slice(guidance[-1], 1 * self.out_channels, self.out_channels)
        gate3_wb_cmb = tf.slice(guidance[-1], 2 * self.out_channels, self.out_channels)
        gate4_wb_cmb = tf.slice(guidance[-1], 3 * self.out_channels, self.out_channels)
        gate5_wb_cmb = tf.slice(guidance[-1], 4 * self.out_channels, self.out_channels)
        gate6_wb_cmb = tf.slice(guidance[-1], 5 * self.out_channels, self.out_channels)
        gate7_wb_cmb = tf.slice(guidance[-1], 6 * self.out_channels, self.out_channels)
        gate8_wb_cmb = tf.slice(guidance[-1], 7 * self.out_channels, self.out_channels)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        self.left_top_pad = keras.layers.ZeroPadding2D(
            (0,0,2,2)
        )
        gate1_wb_cmb = self.left_top_pad(gate1_wb_cmb) #.unsqueeze(1)

        self.center_top_pad = keras.layers.ZeroPadding2D(
            (1,0,2,1)
        )
        gate2_wb_cmb = self.center_top_pad(gate2_wb_cmb) #.unsqueeze(1)

        self.right_top_pad = keras.layers.ZeroPadding2D(
            (2,0,2,0)
        )
        gate3_wb_cmb = self.right_top_pad(gate3_wb_cmb) #.unsqueeze(1)

        # center pad
        self.left_center_pad = keras.layers.ZeroPadding2D(
            (0,1,1,2)
        )
        gate4_wb_cmb = self.left_center_pad(gate4_wb_cmb) #.unsqueeze(1)

        self.right_center_pad = keras.layers.ZeroPadding2D(
            (2,1,1,0))
        gate5_wb_cmb = self.right_center_pad(gate5_wb_cmb) #.unsqueeze(1)

        # bottom pad
        self.left_bottom_pad = keras.layers.ZeroPadding2D((0,2,0,2))
        gate6_wb_cmb = self.left_bottom_pad(gate6_wb_cmb) #.unsqueeze(1)

        self.center_bottom_pad = keras.layers.ZeroPadding2D((1,2,0,1))
        gate7_wb_cmb = self.center_bottom_pad(gate7_wb_cmb) #.unsqueeze(1)

        self.right_bottm_pad = nn.ZeroPad2d((2,2,0,0))
        gate8_wb_cmb = self.right_bottm_pad(gate8_wb_cmb) #.unsqueeze(1)

        gate_wb = tf.concat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                             gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = tf.math.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)

        gate_wb = tf.math.div(gate_wb, abs_weight)
        gate_sum = self.sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, 1:-1, 1:-1, :]

        return gate_wb, gate_sum
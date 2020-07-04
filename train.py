
import argparse

import tensorflow as tf
import tensorflow.keras as keras

from tf_anynet.models.anynet import AnyNet, L1DisparityMaskLoss
from tf_anynet.dataset import DrivingDataset, DrivingTFRecordsDataset


SAMPLES=300
def parse_args():
    parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
    parser.add_argument('--global_max_disp', type=int, default=192, help='Global maximum disparity (pixels)')
    parser.add_argument('--local_max_disps', type=int, nargs='+', default=[12, 3, 3], help='Maximum disparity localized per Disparity Net stage')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
    parser.add_argument('--datapath', default='dataset/',
                        help='datapath')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--train_bsize', type=int, default=6,
                        help='batch size for training (default: 12)')
    parser.add_argument('--test_bsize', type=int, default=4,
                        help='batch size for testing (default: 8)')
    parser.add_argument('--save_path', type=str, default='results/pretrained_anynet',
                        help='the path of saving checkpoints and log')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume path')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
    parser.add_argument('--unet_cond2d_filters', type=int, default=1, help='Initial num Conv2D output filters of Unet feature extractor')
    parser.add_argument('--nblocks', type=int, default=2, help='number of blocks in each conv stage')
    parser.add_argument('--disp_conv3d_filters', type=int, default=4, help='Initial num Conv3D output filters of Disparity Network (multiplied by growth_rate[stage_num])')
    parser.add_argument('--disp_conv3d_layers', type=int, default=4, help='Num Conv3D layers of Disparty Network')
    parser.add_argument('--disp_conv3d_growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
    parser.add_argument('--spn_init_filters', type=int, default=8, help='initial channels for spnet')
    return parser.parse_args()

def main():
    args = parse_args()
    stages = 3 + args.with_spn

    tf.config.experimental_run_functions_eagerly(True)

    ds = DrivingTFRecordsDataset(training=True)
    
    split = int(SAMPLES * .7)
    train_ds = ds.take(split)
    test_ds = ds.take(split)

    

    model = AnyNet(batch_size=args.train_bsize)

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=L1DisparityMaskLoss(
            args.loss_weights,
            stages,
            args.global_max_disp
        )
    )


    model.fit(train_ds, 
        epochs=args.epochs, 
        batch_size=args.train_bsize,
        validation_data=test_ds
        )

if __name__ == '__main__':
    main()

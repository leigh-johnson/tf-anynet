
import argparse
from datetime import datetime
import os

import tensorflow as tf
import tensorflow.keras as keras

from tf_anynet.models.callbacks import DepthMapImageCallback
from tf_anynet.models.anynet_fn import AnyNet
from tf_anynet.models.metrics import (
    L1DisparityMaskLoss,
)
from tf_anynet.dataset import TFRecordsDataset


SAMPLES=2200

def parse_args():
    parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed (training dataset shuffle)')
    parser.add_argument('--global_max_disp', type=int, default=192, help='Global maximum disparity (pixels)')
    parser.add_argument('--local_max_disps', type=int, nargs='+', default=[12, 24, 48], help='Maximum disparity localized per Disparity Net stage')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
    parser.add_argument('--datapath', default='dataset/',
                        help='datapath')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train')
    parser.add_argument('--train_bsize', type=int, default=20,
                        help='batch size for training (default: 6)')
    parser.add_argument('--test_bsize', type=int, default=1,
                        help='batch size for testing (default: 1)')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume path')
    parser.add_argument('--checkpoint', type=str, default=None,
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
    parser.add_argument('--train_ds', type=str, default='flyingthings_train.shard*')
    parser.add_argument('--test_ds', type=str, default='flyingthings_test.shard*')
    return parser.parse_args()

def to_x_y(imgL, imgR, disp):
    return ((imgL, imgR), disp)
def main():
    args = parse_args()
    stages = 3 + args.with_spn

    #tf.config.experimental_run_functions_eagerly(True)

    # ds = TFRecordsDataset('driving.tfrecords.shard*',training=True)
    # ds = ds.map(lambda imgL, imgR, dispL: ((imgL, imgR), dispL))
    
    # train_size = int(0.7 * SAMPLES)

    # #ds = ds.shuffle(args.train_bsize, seed=args.seed, reshuffle_each_iteration=False)

    # train_ds = ds.take(train_size).batch(args.train_bsize)
    
    # test_ds = ds.skip(train_size).batch(args.train_bsize)
    # val_ds = ds.skip(train_size).take(args.train_bsize).batch(args.train_bsize)

    train_ds = TFRecordsDataset(args.train_ds, training=True)\
        .map(to_x_y, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .shuffle(args.train_bsize*4, reshuffle_each_iteration=True)\
        .batch(args.train_bsize,drop_remainder=True)\
        .prefetch(1)

    test_ds  = TFRecordsDataset(args.test_ds, training=True)\
        .map(to_x_y, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .prefetch(1)
    

    val_ds = test_ds.skip(args.train_bsize).take(args.train_bsize)
    test_ds = test_ds.batch(args.train_bsize, drop_remainder=True).prefetch(1)
    
    model_builder = AnyNet(
        batch_size=args.train_bsize, 
        eval_data=val_ds
    )
    
    input_shape = train_ds.element_spec[0][0].shape
    model = model_builder.build(input_shape=input_shape)

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    if args.resume:
        log_dir = f'{args.resume}'

    else:
        log_dir = str(datetime.now()).replace(' ', 'T')
        log_dir = f'./logs/{log_dir}'

    metrics = [
        keras.metrics.RootMeanSquaredError(),
        keras.metrics.MeanAbsolutePercentageError()
        #RootMeanSquaredError(),
        #MeanAbsolutePercentageError()
    ]

    model.compile(
        optimizer=optimizer,
        loss={
            f'disparity-{i}': L1DisparityMaskLoss(
                i,
                args.global_max_disp
                )
            for i in range(0, stages)
        },
        loss_weights={
            f'disparity-{i}': args.loss_weights[i]
            for i in range(0, stages)
        },
        metrics=metrics
    )

    callbacks = [
        DepthMapImageCallback(val_ds, log_dir=log_dir),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir+'/model.{epoch:02d}-{val_loss:.2f}.hdf5',
            #save_best_only=True,
            mode='min',
            save_weights_only=False,
            verbose=1
        )
    ]

    if args.checkpoint:
        model.load_weights(args.checkpoint)

    model.fit(
        train_ds,
        epochs=args.epochs, 
        batch_size=args.train_bsize,
        validation_data=test_ds,
        callbacks=callbacks
    )

if __name__ == '__main__':
    main()


import argparse
from datetime import datetime
import os

import tensorflow as tf
import tensorflow.keras as keras

from tf_anynet.models.callbacks import DepthMapImageCallback
from tf_anynet.models.anynet_fn import AnyNet
from tf_anynet.models.metrics import (
    L1DisparityMaskLoss,
    masked_pixel_ratio
)
from tf_anynet.dataset import TFRecordsDataset, random_crop, center_crop, to_x_y


SAMPLES=2200

def parse_args():
    parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed (training dataset shuffle)')
    parser.add_argument('--global_max_disp', type=int, default=256, help='Global maximum disparity (pixels)')
    parser.add_argument('--local_max_disps', type=int, nargs='+', default=[24, 16, 8], help='Maximum disparity localized per Disparity Net stage')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
    parser.add_argument('--datapath', default='dataset/',
                        help='datapath')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train')
    parser.add_argument('--train_bsize', type=int, default=92,
                        help='batch size for training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='resume path')                        
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--learning_rate_end', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_steps', type=int, default=3000,
                        help='learning rate')                       
    parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
    parser.add_argument('--unet_conv2d_filters', type=int, default=1, help='Initial num Conv2D output filters of Unet feature extractor')
    parser.add_argument('--unet_nblocks', type=int, default=2, help='number of blocks in each conv stage')
    parser.add_argument('--disp_conv3d_filters', type=int, default=4, help='Initial num Conv3D output filters of Disparity Network (multiplied by growth_rate[stage_num])')
    parser.add_argument('--disp_conv3d_layers', type=int, default=4, help='Num Conv3D layers of Disparty Network')
    parser.add_argument('--disp_conv3d_growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
    parser.add_argument('--cspn_conv3d_filters', type=int, default=8, help='initial channels for spnet')
    parser.add_argument('--cspn_conv3d_step', type=int, default=24, help='initial channels for spnet')
    parser.add_argument('--train_ds', type=str, default='flyingthings_train.shard*')
    parser.add_argument('--test_ds', type=str, default='flyingthings_test.shard*')
    #parser.add_argument('--train_ds', type=str, default='driving.shard-[1-5]*')
    #parser.add_argument('--test_ds', type=str, default='driving.shard-0.gz')
    parser.add_argument('--epsilon', type=float, default=1e-07)
    parser.add_argument('--mlflow', action='store_true', help='Initialize MLFlow experiment logging')
    parser.add_argument('--initial_epoch', type=int, default=0, help="Begin epoch counter at this number")
    parser.add_argument('--resume', action='store_true', help='with spn network or not')
    return parser.parse_args()

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
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
    train_cache_file = args.train_ds.split('.')[0]
    train_ds = TFRecordsDataset(args.train_ds, training=True)\
        .map(random_crop, num_parallel_calls=4)\
        .shuffle(args.train_bsize*8, reshuffle_each_iteration=True)\
        .batch(args.train_bsize,drop_remainder=True)\
        .prefetch(3)\
        .take(args.train_bsize*2)

    test_cache_file = args.test_ds.split('.')[0]
    test_ds  = TFRecordsDataset(args.test_ds, training=True)\
        .map(center_crop, num_parallel_calls=4)\
        .batch(args.train_bsize, drop_remainder=True)\
        .prefetch(3)\
        .take(args.train_bsize*2)
    

    val_ds = test_ds.take(1)
    
    model_builder = AnyNet(
        batch_size=args.train_bsize, 
        unet_conv2d_filters=args.unet_conv2d_filters,
        unet_nblocks=args.unet_nblocks,
        cspn_conv3d_filters=args.cspn_conv3d_filters,
        local_max_disps=args.local_max_disps,
        global_max_disp=args.global_max_disp,
        loss_weights=args.loss_weights,
        stages=stages,
    )
    
    input_shape = train_ds.element_spec[0][0].shape
    model = model_builder.build(input_shape=input_shape)

    initial_learning_rate = args.learning_rate
    end_learning_rate = args.learning_rate_end
    decay_steps = args.learning_rate_end
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=args.epsilon)
    
    if args.resume:
        # --checkpoint logs/2020-07-17T02:57:23.979082/model.01-57.78.hdf5
        log_name = args.checkpoint.split('/')
        log_dir = '/'.join(log_name[:2])

    else:
        log_name = str(datetime.now()).replace(' ', 'T')
        log_dir = f'./logs/{log_name}'

    rmse0 = keras.metrics.RootMeanSquaredError(name="rmse_0")
    rmse1 = keras.metrics.RootMeanSquaredError(name="rmse_1")
    rmse2 = keras.metrics.RootMeanSquaredError(name="rmse_2")
    rmse_agg = keras.metrics.RootMeanSquaredError(name="rmse_agg")

    def included_pixel_avg(y_true, y_pred):
        return masked_pixel_ratio(y_true, args.global_max_disp)

    metrics = {
        'disparity-0': [rmse0, rmse_agg],
        'disparity-1': [rmse1, rmse_agg],
        'disparity-2': [rmse2, rmse_agg, included_pixel_avg],
    }

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
        DepthMapImageCallback(val_ds, args.train_bsize, args.train_bsize, frequency=10, log_dir=log_dir),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=5,
            profile_batch='60,70'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            #filepath=log_dir+'/model.{epoch:02d}-{val_loss:.2f}.hdf5',
            filepath=log_dir+'/{epoch:02d}-{val_loss:.2f}.ckpt', #+'/model.{epoch:02d}-{val_loss:.2f}.hdf5'
            save_best_only=True,
            mode='min',
            save_weights_only=True,
            verbose=1
        )
    ]

    if args.checkpoint:
        weights_file = tf.train.latest_checkpoint(args.checkpoint)

        model.load_weights(weights_file)
        #model = tf.keras.models.load_model(args.checkpoint)

    if args.mlflow:
        import mlflow.tensorflow
        from git.repo.base import Repo 

        repo = Repo('.')
        diff = repo.git.diff('HEAD~1')
        f_diff = ['\t'] + diff.splitlines()
        f_diff = '\n\t'.join(f_diff)
        with mlflow.start_run(run_name=log_name):
            mlflow.log_params(vars(args))
            mlflow.tensorflow.autolog(every_n_iter=10)
            mlflow.set_tag('mlflow.note.content', f_diff)
            mlflow.set_tag('tensorboard', log_name)
            model.fit(
                train_ds,
                epochs=args.epochs, 
                batch_size=args.train_bsize,
                validation_data=test_ds,
                callbacks=callbacks,
                initial_epoch=args.initial_epoch
            )
    else:
        model.fit(
            train_ds,
            epochs=args.epochs, 
            batch_size=args.train_bsize,
            validation_data=test_ds,
            callbacks=callbacks,
            initial_epoch=args.initial_epoch
        )
if __name__ == '__main__':
    main()

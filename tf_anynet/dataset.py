import os
import pathlib
import random

import numpy as np
import tensorflow as tf

CROP_H = 256
CROP_W = 512

FULL_W = 960
FULL_H = 540

def to_normalized_x_y(ds):
    print('Loading x1')
    x1 = ds.map(to_x1, deterministic=True, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print('Loading x2')
    x2 = ds.map(to_x2, deterministic=True, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print('Loading y')
    y = ds.map(to_y, deterministic=True, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization(
        axis=-1
    )
    normalizer.adapt(x1)
    print('Fit normalizer to x1')
    print(f'mean: {normalizer.mean} var: {normalizer.variance} count: {normalizer.count}')
    normalizer.adapt(x2)
    print('Fit normalizer to x2')
    print(f'mean: {normalizer.mean} var: {normalizer.variance} count: {normalizer.count}')
    x1 = normalizer(x1)
    x2 = normalizer(x2)
    return ((x1, x2), y)
    
def to_y(example_proto):
    tfrecord = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    disp_img = tf.reshape(
        tf.io.decode_raw(tfrecord['disp_raw'], tf.float32),
        [FULL_H, FULL_W, 1]
    )
    return disp_img

def to_x1(example_proto):
    tfrecord = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

    left_img = tf.io.decode_raw(tfrecord['left_img_raw'], tf.float32)
    left_img = tf.reshape(left_img, [FULL_H, FULL_W, 3])

    return left_img

def to_x2(example_proto):
    tfrecord = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    right_img = tf.io.decode_raw(tfrecord['right_img_raw'], tf.float32)
    right_img = tf.reshape(right_img, [FULL_H, FULL_W, 3])
    return right_img

def to_x_y(example_proto):
    tfrecord = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

    left_img = tf.io.decode_raw(tfrecord['left_img_raw'], tf.float32)
    left_img = tf.reshape(left_img, [FULL_H, FULL_W, 3])

    right_img = tf.io.decode_raw(tfrecord['right_img_raw'], tf.float32)
    right_img = tf.reshape(right_img, [FULL_H, FULL_W, 3])

    disp_img = tf.reshape(
        tf.io.decode_raw(tfrecord['disp_raw'], tf.float32),
        [FULL_H, FULL_W, 1]
    ) 
    return (left_img, right_img, disp_img)

def center_crop(left_img, right_img, disp_img, upsample=False):

    offset_height = (FULL_H - CROP_H) // 2
    offset_width = (FULL_W - CROP_W) // 2
    left_img = tf.image.crop_to_bounding_box(left_img, offset_height, offset_width, CROP_H, CROP_W)
    right_img = tf.image.crop_to_bounding_box(right_img, offset_height, offset_width, CROP_H, CROP_W)
    disp_img = tf.image.crop_to_bounding_box(disp_img, offset_height, offset_width, CROP_H, CROP_W)
    if upsample:
        return (
            tf.image.resize(left_img, [FULL_H, FULL_W]),
            tf.image.resize(right_img, [FULL_H, FULL_W]),
            tf.image.resize(disp_img, [FULL_H, FULL_W])
        )
    else:
        return ((left_img, right_img), disp_img)

def random_crop(left_img, right_img, disp_img, upsample=False):

    # 3D float32 [0, 1]
    # left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    # right_img = tf.image.convert_image_dtype(right_img, tf.float32)

    # random crop
    x1 = random.randint(0, FULL_W - CROP_W)
    y1 = random.randint(0, FULL_H - CROP_H)

    left_img = tf.image.crop_to_bounding_box(
        left_img, y1, x1, CROP_H, CROP_W
    )
    right_img = tf.image.crop_to_bounding_box(
        right_img, y1, x1, CROP_H, CROP_W
    )

    disp_img = tf.image.crop_to_bounding_box(
        disp_img, y1, x1, CROP_H, CROP_W
    )
    if upsample:
        return (
            tf.image.resize(left_img, [FULL_H, FULL_W]),
            tf.image.resize(right_img, [FULL_H, FULL_W]),
            tf.image.resize(disp_img, [FULL_H, FULL_W])
        )
    else:
        return ((left_img, right_img), disp_img)


class FlyingThingsTrainDataset(tf.data.Dataset):
    DATA_ROOT =  '/home/leigh/torrents/flyingthings3d/'


    def __new__(self, training=False):
        self.training = training
        driving_root = pathlib.Path(self.DATA_ROOT)

        img_files_l = tf.data.Dataset.list_files(
            str(driving_root/'frames_cleanpass/TRAIN/*/*/left/*.png'),
            shuffle=False
            )
        img_files_r = tf.data.Dataset.list_files(str(driving_root/'frames_cleanpass/TRAIN/*/*/right/*.png'),
            shuffle=False)
        disp_files_l = tf.data.Dataset.list_files(str(driving_root/'disparity/TRAIN/*/*/left/*.pfm'),
            shuffle=False)

        ds = tf.data.Dataset.zip(
            (img_files_l, img_files_r, disp_files_l)
        )
        return ds

class FlyingThingsTestDataset(tf.data.Dataset):
    DATA_ROOT =  '/home/leigh/torrents/flyingthings3d/'


    def __new__(self, training=False):
        self.training = training
        driving_root = pathlib.Path(self.DATA_ROOT)

        img_files_l = tf.data.Dataset.list_files(
            str(driving_root/'frames_cleanpass/TEST/*/*/left/*.png'),
            shuffle=False
            )
        img_files_r = tf.data.Dataset.list_files(str(driving_root/'frames_cleanpass/TEST/*/*/right/*.png'),
            shuffle=False)
        disp_files_l = tf.data.Dataset.list_files(str(driving_root/'disparity/TEST/*/*/left/*.pfm'),
            shuffle=False)

        ds = tf.data.Dataset.zip(
            (img_files_l, img_files_r, disp_files_l)
        )
        return ds

class DrivingDataset(tf.data.Dataset):

    DATA_ROOT =  '/home/leigh/torrents/driving'


    def __new__(self, training=False):
        self.training = training
        driving_root = pathlib.Path(self.DATA_ROOT)

        img_files_l = tf.data.Dataset.list_files(
            str(driving_root/'frames_cleanpass/15mm_focallength/*/*/left/*.png'),
            shuffle=False
            )
        img_files_r = tf.data.Dataset.list_files(str(driving_root/'frames_cleanpass/15mm_focallength/*/*/right/*.png'),
            shuffle=False)
        disp_files_l = tf.data.Dataset.list_files(str(driving_root/'disparity/15mm_focallength/*/*/left/*.pfm'),
            shuffle=False)

        ds = tf.data.Dataset.zip(
            (img_files_l, img_files_r, disp_files_l)
        )
        return ds

# Create a description of the features.
FEATURE_DESCRIPTION = {
    'uid': tf.io.FixedLenFeature([], tf.string),
    'disp_raw': tf.io.FixedLenFeature([], tf.string),
    'disp_path': tf.io.FixedLenFeature([], tf.string),
    'left_img_raw': tf.io.FixedLenFeature([], tf.string),
    'left_img_path': tf.io.FixedLenFeature([], tf.string),
    'right_img_raw': tf.io.FixedLenFeature([], tf.string),
    'right_img_path': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'img_channels': tf.io.FixedLenFeature([], tf.int64),
    'disp_channels': tf.io.FixedLenFeature([], tf.int64)
}

def deserialize_tfrecord(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

class TFRecordsDataset(tf.data.Dataset):

    DATA_ROOT =  'data/'


    def __new__(self, pattern, calc_normalize=False):
        data_root = pathlib.Path(self.DATA_ROOT)

        files = tf.data.Dataset.list_files(
            str(data_root/pattern),
            shuffle=False
            )
        
        ds = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=24)
        
        if calc_normalize:
            return ds.apply(to_normalized_x_y)
        else:
            return ds\
            .map(to_x_y, num_parallel_calls=4)

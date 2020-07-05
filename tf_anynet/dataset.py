import os
import pathlib
import random

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


CROP_H = 256
CROP_W = 512

FULL_W = 960
FULL_H = 540

def random_crop(tfrecord, training=True):


    left_img = tf.image.decode_png(tfrecord['left_img_raw'], channels=3)
    right_img = tf.image.decode_png(tfrecord['right_img_raw'], channels=3)

    # 3D float32 [0, 1]
    left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    right_img = tf.image.convert_image_dtype(right_img, tf.float32)
    
    disp_img = tf.reshape(
        tf.io.decode_raw(tfrecord['disp_raw'], tf.float32),
        [FULL_H, FULL_W, 1]
    )
    # crop to TRAIN_H, TRAIN_W
    if training:

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
        
    return (
        tf.image.resize(left_img, [FULL_H, FULL_W]),
        tf.image.resize(right_img, [FULL_H, FULL_W]),
        tf.image.resize(disp_img, [FULL_H, FULL_W])
    )

class FlyingThings3DDataset(tf.data.Dataset):
    pass

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

class DrivingTFRecordsDataset(tf.data.Dataset):

    DATA_ROOT =  'data/'


    def __new__(self, training=False):
        self.training = training
        data_root = pathlib.Path(self.DATA_ROOT)

        files = tf.data.Dataset.list_files(
            str(data_root/'driving.tfrecords.shard*'),
            shuffle=False
            )
        
        ds = tf.data.TFRecordDataset(files, compression_type='GZIP')
        return ds\
            .map(deserialize_tfrecord)\
            .map(random_crop)

if __name__ == '__main__':
    ds = DrivingTFRecordsDataset(training=True)

    import pdb; pdb.set_trace()
    print(len(ds))
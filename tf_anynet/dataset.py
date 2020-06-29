import os
import pathlib
import random

import numpy as np
import tensorflow as tf


CROP_H = 256
CROP_W = 512

FULL_W = 960
FULL_H = 540

def decode_img(img_l, img_r, training=True):
    left_data = tf.io.read_file(img_l)
    right_data = tf.io.read_file(img_r)
    # 3D uint8 tensor

    left_img = tf.image.decode_png(left_data, channels=3)
    right_img = tf.image.decode_png(right_data, channels=3)

    # 3D float32 [0, 1]
    left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    right_img = tf.image.convert_image_dtype(right_img, tf.float32)

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
    return (
        tf.image.resize(left_img, [FULL_H, FULL_W]),
        tf.image.resize(right_img, [FULL_H, FULL_W]),
    )

# def process_disp_map(disp_map_file):
#     disp_data = tf.io.read_file(disp_map_file)
    
#     # #disp_img = tf.strings.unicode_decode(disp_data, 'UTF-8')
#     # disp_img = tf.strings.bytes_split(disp_data, 'UTF-8')
#     # disp_img = disp_img[18:] # disp_img[18:]
#     # disp_img = tf.reshape(disp_img, shape=(4,))
#     # #disp_img = tf.io.decode_raw(disp_img, tf.float32)
#     # #disp_img = tf.reshape(disp_img, shape=(FULL_H, FULL_W, 1))
    
#     return data

def process_files(img_l, img_r, disp_l, training=True):
    return decode_img(img_l, img_r, training=training), disp_l # process_disp_map(disp_l)


class FlyingThings3DDataset(tf.data.Dataset):
    pass

class DrivingDataset(tf.data.Dataset):

    DRIVING_ROOT =  '/home/leigh/torrents/driving'


    def __new__(self, training=False):
        self.training = training
        driving_root = pathlib.Path(self.DRIVING_ROOT)

        img_files_l = tf.data.Dataset.list_files(
            str(driving_root/'frames_cleanpass/15mm_focallength/*/*/left/*.png'),
            shuffle=False
            )\
            
            #.map(tf.io.read_file)
        img_files_r = tf.data.Dataset.list_files(str(driving_root/'frames_cleanpass/15mm_focallength/*/*/right/*.png'),
            shuffle=False)\
            #.map(tf.io.read_file)
        disp_files_l = tf.data.Dataset.list_files(str(driving_root/'disparity/15mm_focallength/*/*/left/*.pfm'),
            shuffle=False)\
            #.map(tf.io.read_file)

        ds = tf.data.Dataset.zip(
            (img_files_l, img_files_r, disp_files_l)
        )
        return ds
        # return ds.map(process_files, num_parallel_calls=1)
    
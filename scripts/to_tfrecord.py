import click

import numpy as np
import tensorflow as tf

from tf_anynet.dataset import DrivingDataset, FlyingThingsTestDataset, FlyingThingsTrainDataset

DATASETS = {
  'driving': DrivingDataset,
  'flyingthings_train': FlyingThingsTrainDataset,
  'flyingthings_test': FlyingThingsTestDataset
}

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class IDMismatch(Exception):
  pass

def parse_uid(
    left_img_path,
    right_img_path, 
    disp_img_path
  ):
  luid = tf.strings.split(
    tf.strings.split(left_img_path, sep='/')[-1],
    sep='.'
  )[0]
  ruid = tf.strings.split(
    tf.strings.split(right_img_path, sep='/')[-1],
    sep='.'
  )[0]

  duid = tf.strings.split(
    tf.strings.split(disp_img_path, sep='/')[-1],
    sep='.'
  )[0]
  
  if luid == ruid and luid == duid:
    return luid
  else:
    raise IDMismatch(f'Mismatched file ids! \n\
      {left_img_path} \n\
      {right_img_path} \n\
      {disp_img_path}'
    )

def decode_pfm_dims(disp_metadata):
  ''' 
    From: https://linux.die.net/man/5/pfm
  '''
  return (int(x) for x in disp_metadata[1].strip().split(b' '))

def decode_pfm_channels(disp_metadata):
  ''' 
    From: https://linux.die.net/man/5/pfm

    The identifier line contains the characters 'PF' or 'Pf'. 
    PF means it's a color PFM. Pf means it's a grayscale PFM.

  '''
  channels = disp_metadata[0].strip()

  if channels != b'Pf' and channels != b'PF':
    raise Exception(f"Expected b'Pf' or b'PF' bytes for channels but got {channels}")
  return 3 if channels == b'PF' else 1

def decode_pfm_endian(disp_metadata):
  if disp_metadata[-1].strip() == b'-1.0':
    return '<'
  elif disp_metadata[-1].strip() == b'1.0':
    return '>'
  raise Exception(f"Expected b'-1.0' or b'1.0' for endianess but got {disp_metadata[-1].strip()}")

def parse_disp(disp_data):
  raw_bytes = disp_data.numpy().split(b'\n')
  disp_metadata = raw_bytes[:3]
  raw_bytes = b'\n'.join(raw_bytes[3:])
  
  width, height = decode_pfm_dims(disp_metadata)
  channels = decode_pfm_channels(disp_metadata)
  endian = decode_pfm_endian(disp_metadata)
  
  dt = np.dtype(np.float32)
  dt = dt.newbyteorder(endian)
  
  disp_img = np.frombuffer(raw_bytes, dtype=dt)
  
  disp_img = np.reshape(disp_img, (height, width, channels))

  disp_img = np.flipud(disp_img)
  return disp_img

def create_tfexample(
   left_img_path,
   right_img_path, 
   disp_img_path
  ):

  left_img = tf.io.read_file(left_img_path)
  left_img = tf.image.decode_png(left_img)
  image_shape = left_img.shape

  left_img = tf.image.convert_image_dtype(left_img, tf.float32)
  left_img = left_img.numpy().tobytes()

  right_img = tf.io.read_file(right_img_path)
  right_img = tf.image.decode_png(right_img)
  right_img = tf.image.convert_image_dtype(right_img, tf.float32)
  right_img = right_img.numpy().tobytes()

  disp_data = tf.io.read_file(disp_img_path)

  disp_img = parse_disp(disp_data)

  # this was a sanity check
  # disp_png_path = tf.strings.split(disp_img_path, sep='.')[0].numpy().decode('utf-8')

  # import imageio
  # disp_png_path = disp_png_path + '.png'
  # imageio.imwrite(disp_png_path, disp_img)

  uid = parse_uid(left_img_path, right_img_path, disp_img_path)
  feature = {
      'uid': _bytes_feature(uid),
      'disp_raw': _bytes_feature(disp_img.tobytes()),
      'disp_path': _bytes_feature(disp_img_path),
      'left_img_raw':  _bytes_feature(left_img),
      'left_img_path': _bytes_feature(left_img_path),
      'right_img_raw': _bytes_feature(right_img),
      'right_img_path': _bytes_feature(right_img_path),
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'img_channels': _int64_feature(image_shape[2]),
      'disp_channels': _int64_feature(disp_img.shape[-1])
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

@click.group()
def cli():
  pass

@cli.command()
@click.option('--dataset', default='driving')
@click.option('-f', '--file', default='data/driving.tfrecords.gz')
def pack(dataset, file):

  ds = DATASETS[dataset]()

  writer_opts = tf.io.TFRecordOptions(
    compression_type="GZIP"
  )

  ds_length = sum(1 for x in ds.as_numpy_iterator())

  with tf.io.TFRecordWriter(file, options=writer_opts) as writer:  
    for i, (left_img_path, right_img_path, disp_img_path) in enumerate(ds.as_numpy_iterator()):
      tf_example = create_tfexample(left_img_path, right_img_path, disp_img_path)
      writer.write(tf_example.SerializeToString())
      if i % 100 == 0:
        print(f'Finished {i} / {ds_length}')

@click.option('-f', '--file', default='data/flyingthings_test.tfrecords.gz')
@click.option('--shards', default=8)
@cli.command()
def shard(file, shards):
  raw_dataset = tf.data.TFRecordDataset(file, compression_type="GZIP")

  for i in range(shards):
      outfile = file.split('.')[0] + f'.shard-{i}.gz'
      writer = tf.data.experimental.TFRecordWriter(outfile.format(shard=i), compression_type="GZIP")
      writer.write(raw_dataset.shard(shards, i))
      print(f'Finished {i+1}/{shards}')

if __name__ == '__main__':
  cli()
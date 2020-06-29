import tensorflow as tf

from tf_anynet.dataset import DrivingDataset

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

def parse_disp(disp_data):
  return disp_data.numpy()[18]
def create_tfexample(
   left_img_path,
   right_img_path, 
   disp_image_path
  ):

  left_img = tf.io.read_file(left_img_path)
  right_img = tf.io.read_file(right_img_path)
  disp_data = tf.io.read_file(disp_image_path)

  disp_raw = parse_disp(disp_data)

  uid = parse_uid(left_img_path, right_img_path, disp_image_path)
  image_shape = tf.image.decode_png(left_img).shape
  feature = {
      'uid': _bytes_feature(uid),
      'disp_raw': _float_feature(disp_raw),
      'disp_path': _bytes_feature(disp_image_path),
      'left_img_raw':  _bytes_feature(left_img),
      'left_img_path': _bytes_feature(left_img_path),
      'right_img_raw': _bytes_feature(right_img),
      'right_img_path': _bytes_feature(right_img_path),
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'img_channels': _int64_feature(image_shape[2]),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def main():
  ds = DrivingDataset()
  record_path='data/driving.tfrecords'

  writer_opts = tf.io.TFRecordOptions(
    compression_type="GZIP"
  )
  ds_length = sum(1 for x in ds.as_numpy_iterator())

  with tf.io.TFRecordWriter(record_path, options=writer_opts) as writer:  
    for i, (left_img_path, right_img_path, disp_image_path) in enumerate(ds.as_numpy_iterator()):
      tf_example = create_tfexample(left_img_path, right_img_path, disp_image_path)
      writer.write(tf_example.SerializeToString())
      if i % 100 == 0:
        print(f'Finished {i} / {ds_length}')
  #return ds.map(create_tfexample)
  #\
    #.map(create_tfrecord)

def shard():
  raw_dataset = tf.data.TFRecordDataset("data/driving.tfrecords.gz", compression_type="GZIP")
  shards = 10

  for i in range(shards):
      writer = tf.data.experimental.TFRecordWriter(f"data/driving.tfrecords.shard-{i}.gz", compression_type="GZIP")
      writer.write(raw_dataset.shard(shards, i))
if __name__ == '__main__':
  shard()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tarfile
import io
import PIL.Image

from six.moves import urllib

import tensorflow as tf

DATA_URL = \
  'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
DEFAULT_DATA_DIR = '/tmp/pascal_voc'
DEFAULT_RECORD_DIR = '/tmp/pascal_voc'


def maybe_download_pascal_voc(data_dir, force=False):
  if not os.path.exists(data_dir):
    os.makedev(data_dir)

  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(data_dir, filename)

  if not os.path.exists(filepath) or force:
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath).extractall(data_dir)


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def dict_to_example(data):
  img_path = os.path.join(data['img_dir'], data['image'] + '.jpg')
  gt_path = os.path.join(data['gt_dir'], data['image'] + '.png')
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_img = fid.read()
  with tf.gfile.GFile(gt_path, 'rb') as fid:
    encoded_gt = fid.read()
  encoded_img_io = io.BytesIO(encoded_img)
  encoded_gt_io = io.BytesIO(encoded_gt)
  image = PIL.Image.open(encoded_img_io)
  ground_truth = PIL.Image.open(encoded_gt_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG', image.format)
  if ground_truth.format != 'PNG':
    raise ValueError('Ground truth format not PNG', ground_truth.format)
  if image.size != ground_truth.size:
    raise ValueError(
        'Train image and ground truth image should be of the same size',
        image.size, ground_truth.size)

  height, width = image.size
  channels = len(image.mode)

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'image/channels': _int64_feature(channels),
    'image/encoded': _bytes_feature(encoded_img),
    'image/format': _bytes_feature(image.format.encode('utf8')),
    'ground_truth/encoded': _bytes_feature(encoded_gt),
    'ground_truth/format': _bytes_feature(ground_truth.format.encode('utf8'))}))
  return example


def label_map_dict_gen(data_dir, data_set='train'):
  if data_set not in ['train', 'trainval', 'val']:
    raise ValueError(
        'data_set must be one of \'train\', \'trainval\' or \'val\'', data_set)
  data_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
  label_map_path = os.path.join(data_dir, 'ImageSets', 'Segmentation')
  list_path = os.path.join(label_map_path, data_set + '.txt')

  img_dir = os.path.join(data_dir, 'JPEGImages')
  gt_dir = os.path.join(data_dir, 'SegmentationClass')

  dict = {}
  dict['img_dir'] = img_dir
  dict['gt_dir'] = gt_dir

  with tf.gfile.GFile(list_path, 'r') as fid:
    while True:
      line = fid.readline()
      if not line:
        break
      dict['image'] = line.rstrip()
      yield dict


def prepare_pascal_voc(data_dir, out_dir, force=False):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  train_record = 'train.record'
  val_record = 'val.record'

  if not (os.path.exists(os.path.join(data_dir, train_record)) and
            os.path.exists(os.path.join(data_dir, val_record))) \
      or force:

    maybe_download_pascal_voc(data_dir, force=force)

    train_writer = tf.python_io.TFRecordWriter(
        os.path.join(out_dir, train_record))
    for data in label_map_dict_gen(data_dir, 'train'):
      example = dict_to_example(data)
      train_writer.write(example.SerializeToString())
    train_writer.close()

    val_writer = tf.python_io.TFRecordWriter(
        os.path.join(out_dir, val_record))
    for data in label_map_dict_gen(data_dir, 'val'):
      example = dict_to_example(data)
      val_writer.write(example.SerializeToString())
      val_writer.close()


def preprocess_images(image, ground_truth, out_shape, is_training,
                      num_classes, ignore_label=255):
  height = out_shape[0]
  width = out_shape[1]
  depth_i = image.shape.as_list()[2]
  depth_gt = ground_truth.shape.as_list()[2]

  if is_training:
    # ignore_label needs to be subtracted and later added due to 0 padding.
    # ground_truth -= ignore_label

    combined = tf.concat([image, ground_truth], axis=2)

    combined = tf.image.resize_image_with_crop_or_pad(
        combined, height + 72, width + 72)

    combined = tf.random_crop(combined, [height, width, depth_i + depth_gt])

    combined = tf.image.random_flip_left_right(combined)

    image = combined[:, :, :depth_i]
    ground_truth = combined[:, :, depth_i:] #+ ignore_label

  else:
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    ground_truth = tf.image.resize_image_with_crop_or_pad(
        ground_truth, height, width)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = map_ground_truth(ground_truth, num_classes)
  return image, label


def map_ground_truth(ground_truth, num_classes):
  grayscale = tf.image.rgb_to_grayscale(ground_truth)
  grayscale = tf.squeeze(grayscale)
  label = tf.one_hot(grayscale, num_classes)
  return label


def pascal_voc_input_fn(is_training, num_epochs, batch_size, num_classes,
                        record_dir=DEFAULT_RECORD_DIR,
                        data_dir=DEFAULT_DATA_DIR):
  prepare_pascal_voc(data_dir, record_dir)

  def get_filenames():
    if is_training:
      return os.path.join(record_dir, 'train.record')
    else:
      return os.path.join(record_dir, 'val.record')

  height, width, channels = (500, 500, 3)
  out_shape = [height, width, channels]

  file_names = get_filenames()
  data_set = tf.contrib.data.TFRecordDataset(file_names)

  def dataset_parser(record):
    keys_to_features = {
      # 'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
      # 'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
      # 'image/channels': tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value=''),
      'ground_truth/encoded': tf.FixedLenFeature((), tf.string,
                                                 default_value=''),
      'ground_truth/format': tf.FixedLenFeature((), tf.string,
                                                default_value='')}
    parsed = tf.parse_single_example(record, keys_to_features)

    def decode_image(bytes, out_shape, channels=3, format=tf.constant('JPEG')):
      decoded = tf.cond(tf.equal(format, tf.constant('JPEG')),
                        lambda: tf.image.decode_jpeg(bytes, channels=channels),
                        lambda: tf.image.decode_png(bytes, channels=channels))
      return decoded

    image = decode_image(bytes=parsed['image/encoded'],
                         out_shape=out_shape,
                         channels=channels,
                         format=parsed['image/format'])

    gt = decode_image(bytes=parsed['ground_truth/encoded'],
                      out_shape=out_shape,
                      channels=channels,
                      format=parsed['ground_truth/format'])

    return preprocess_images(
        image, gt, [500, 500, channels], is_training, num_classes)

  data_set = data_set.map(lambda value: dataset_parser(value))
  data_set = data_set.shuffle(buffer_size=10000)
  data_set = data_set.repeat(num_epochs)

  iterator = data_set.batch(batch_size).make_one_shot_iterator()
  images, labels = iterator.get_next()
  images = tf.reshape(images, [batch_size, height, width, channels])
  labels = tf.reshape(labels, [batch_size, height, width, num_classes])

  return images, labels


# Test
if __name__ == '__main__':
  prepare_pascal_voc(DEFAULT_DATA_DIR, DEFAULT_RECORD_DIR)

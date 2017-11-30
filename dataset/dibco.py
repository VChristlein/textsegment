from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from utils.data import maybe_download, dict_to_example, get_label_map_dict
from utils.image_processing import preprocess, scale, inv_preprocess, \
  random_rotate, get_gt_img, map_ground_truth

DATA_URL = 'https://www.dropbox.com/s/62pps8pi5jfqzxg/dibco.zip?dl=1'
DEFAULT_DATA_DIR = '/tmp/dibco'
DEFAULT_RECORD_DIR = '/tmp/dibco'
DATA_EXTRACTED_DIR = 'dibco'
IMG_MEAN = [196.48484802, 188.59724426, 170.53767395]


def prepare_dibco(data_dir=DEFAULT_DATA_DIR,
                  out_dir=DEFAULT_RECORD_DIR,
                  force=False):
  """ Downloads and extracts dibco dataset and its annotation data. """
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  train_record = 'train.record'
  val_record = 'val.record'

  if not (os.path.exists(os.path.join(data_dir, train_record)) and
            os.path.exists(os.path.join(data_dir, val_record))) \
      or force:
    maybe_download(DATA_URL, data_dir, force=force)

    data_dir = os.path.join(data_dir, DATA_EXTRACTED_DIR)

    train_writer = tf.python_io.TFRecordWriter(
      os.path.join(out_dir, train_record))
    for data in get_label_map_dict(data_dir,
                                   os.path.join(data_dir, 'train.txt')):
      example = dict_to_example(data)
      train_writer.write(example.SerializeToString())
    train_writer.close()

    val_writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, val_record))
    for data in get_label_map_dict(data_dir,
                                   os.path.join(data_dir, 'test.txt')):
      example = dict_to_example(data)
      val_writer.write(example.SerializeToString())
      val_writer.close()


def get_dibco_palette():
  return tf.constant(
    [
      [255],  # Background
      [0],  # Writing
    ],
    dtype=tf.int32)


def dibco_input_fn(is_training,
                   num_epochs=1,
                   batch_size=1,
                   img_scale_factor=1,
                   label_size=None,
                   buffer_size=200,
                   record_dir=DEFAULT_RECORD_DIR,
                   data_dir=DEFAULT_DATA_DIR):
  prepare_dibco(data_dir, record_dir)

  height, width, channels_img, channels_gt = (250, 250, 3, 1)
  out_height = int(img_scale_factor * height)
  out_width = int(img_scale_factor * width)
  if label_size is None:
    label_size = (out_height, out_width)

  file_names = os.path.join(
    record_dir, 'train.record' if is_training else 'val.record')
  data_set = tf.contrib.data.TFRecordDataset(file_names)

  def dataset_parser(record):
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'ground_truth/encoded': tf.FixedLenFeature(
        (), tf.string, default_value='')
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=channels_img)

    gt = tf.image.decode_png(
      parsed['ground_truth/encoded'], channels=channels_gt)

    image, gt = preprocess(image, gt, height, width, IMG_MEAN, is_training)
    image = scale(image, scale_factor=img_scale_factor)
    gt = scale(gt, out_size=label_size)
    gt = map_ground_truth(gt, get_dibco_palette())
    image, gt = random_rotate(image, gt)
    return image, gt

  data_set = data_set.map(lambda record: dataset_parser(record))
  data_set = data_set.shuffle(buffer_size=buffer_size)
  data_set = data_set.repeat(num_epochs)

  iterator = data_set.batch(batch_size).make_one_shot_iterator()
  images, labels = iterator.get_next()
  labels = tf.image.resize_nearest_neighbor(labels, label_size)
  images = tf.reshape(images, [batch_size, out_width, out_height, channels_img])

  labels = tf.reshape(
    labels, [batch_size, label_size[0], label_size[1], channels_gt])

  if is_training:
    mode_str = 'train'
  else:
    mode_str = 'eval'

  tf.summary.image(mode_str + '/original', inv_preprocess(images, IMG_MEAN),
                   max_outputs=6)
  tf.summary.image(mode_str + '/ground_truth',
                   get_gt_img(tf.squeeze(labels, axis=3), get_dibco_palette()),
                   max_outputs=6)
  return images, labels


if __name__ == '__main__':
  prepare_dibco(force=True)

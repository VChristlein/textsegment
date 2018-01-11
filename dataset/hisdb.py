from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import io
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

import tensorflow as tf

from utils.data import maybe_download, dict_to_example, get_label_map_dict
from utils.tf_image_processing import preprocess, scale, inv_preprocess, \
  random_rotate, get_gt_img, map_ground_truth


def get_hisdb_meta_data():
  return {
    'url': 'https://www.dropbox.com/s/tc140kgb2k57jen/hisdb.zip?dl=1',
    'img_mean': [159.09439087, 150.34194946, 131.70729065],
    'num_classes': 2,
    'num_img_train': 120,
    'num_img_val': 18,
    'img_channels': 3,
    'gt_channels': 1,
    'default_img_height': 250,
    'default_img_width': 250,
  }


DEFAULT_DATA_DIR = '/tmp/hisdb'
DATA_EXTRACTED_DIR = 'hisdb'


def prepare_hisdb(data_dir=DEFAULT_DATA_DIR,
                  out_dir=None,
                  force=False):
  """ Downloads and extracts dibco dataset and its annotation data. """
  meta_data = get_hisdb_meta_data()

  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  if out_dir is None:
    out_dir = data_dir
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  train_record = 'train.record'
  val_record = 'val.record'

  if not (os.path.exists(os.path.join(data_dir, train_record)) and
          os.path.exists(os.path.join(data_dir, val_record))) or force:
    maybe_download(get_hisdb_meta_data()['url'], data_dir, force=force)

    data_dir = os.path.join(data_dir, DATA_EXTRACTED_DIR)

    train_writer = tf.python_io.TFRecordWriter(
      os.path.join(out_dir, train_record))
    for data in get_label_map_dict(
        data_dir, os.path.join(data_dir, 'train.txt')):
      example = dict_to_example(data, gt_fn=parse_xml)
      train_writer.write(example.SerializeToString())
    train_writer.close()

    val_writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, val_record))
    for data in get_label_map_dict(
        data_dir, os.path.join(data_dir, 'test.txt')):
      example = dict_to_example(data, gt_fn=parse_xml)
      val_writer.write(example.SerializeToString())
    val_writer.close()
    print()

  return meta_data


def get_hisdb_palette():
  return tf.constant(
    [
      [255],  # Background
      [0],  # Writing
    ],
    dtype=tf.int32)


def hisdb_input_fn(is_training,
                   num_epochs=1,
                   batch_size=1,
                   img_size=250,
                   img_scale_factor=1,
                   label_size=None,
                   buffer_size=200,
                   data_dir=DEFAULT_DATA_DIR):
  if isinstance(img_size, tuple):
    height, width = img_size
  else:
    height = width = img_size
  channels_img, channels_gt = (3, 1)
  out_height = int(img_scale_factor * height)
  out_width = int(img_scale_factor * width)
  if label_size is None:
    label_size = (out_height, out_width)

  record = os.path.join(
    data_dir, 'train.record' if is_training else 'val.record')
  if not os.path.exists(record):
    raise ValueError('TFRecord not found: {}.'.format(record) + \
                     'Did you download it `using prepare_hisdb()`?')
  data_set = tf.data.TFRecordDataset(record)

  mean = get_hisdb_meta_data()['img_mean']

  def dataset_parser(record):
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/size': tf.FixedLenFeature((2), tf.int64, default_value=[0, 0]),
      'ground_truth/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
      'ground_truth/size':
        tf.FixedLenFeature((2), tf.int64, default_value=[0, 0]),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=channels_img)

    gt = tf.image.decode_png(
      parsed['ground_truth/encoded'], channels=channels_gt)

    out_size = (height, width)

    image, gt = preprocess(image, gt, out_size, mean, is_training)
    image = scale(image, scale_factor=img_scale_factor)
    gt = scale(gt, out_size=label_size)
    gt = map_ground_truth(gt, get_hisdb_palette())
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

  tf.summary.image(name=mode_str + '/original',
                   tensor=inv_preprocess(images, mean),
                   max_outputs=6)
  tf.summary.image(name=mode_str + '/ground_truth',
                   tensor=get_gt_img(
                     tf.squeeze(labels, axis=3), get_hisdb_palette()),
                   max_outputs=6)
  return images, labels


def parse_xml(path):
  def get_namespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''

  def get_points(root):
    namespace = get_namespace(root)
    for el in root.findall(
        './{0}Page/{0}TextRegion/{0}TextLine/{0}Coords'.format(namespace)):
      yield [tuple(map(int, p.split(','))) \
             for p in el.attrib['points'].split(' ')]

  def get_page_size(root):
    namespace = get_namespace(root)
    el = root.find('./{0}Page'.format(namespace))
    return int(el.attrib['imageWidth']), int(el.attrib['imageHeight'])

  root = ET.parse(path).getroot()

  img = Image.new('L', get_page_size(root), 255)
  for polygon in get_points(root):
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)
  encoded_img_io = io.BytesIO()
  img.save(encoded_img_io, 'PNG')
  return encoded_img_io.getvalue()


if __name__ == '__main__':
  prepare_hisdb(force=False)

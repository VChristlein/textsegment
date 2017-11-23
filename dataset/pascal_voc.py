from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from utils.data import maybe_download, dict_to_example, get_label_map_dict
from utils.image_processing import preprocess, scale, inv_preprocess, \
  random_rotate

DATA_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
DATA_EXTRACTED_DIR = os.path.join('VOCdevkit', 'VOC2012')
DATA_URL_AUG = 'https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1'
DATA_AUG_EXTRACTED_DIR = 'SegmentationClassAug'
DEFAULT_DATA_DIR = '/tmp/pascal_voc'
DEFAULT_RECORD_DIR = '/tmp/pascal_voc'
IMG_MEAN = [104.00698793, 116.66876762, 122.67891434]


def prepare_pascal_voc(data_dir=DEFAULT_DATA_DIR,
                       out_dir=DEFAULT_RECORD_DIR,
                       force=False):
  """ Downloads and extracts pascal voc and its annotation data. """
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
    maybe_download(DATA_URL_AUG, data_dir, force=force)
    aug_target_dir = os.path.join(data_dir, DATA_EXTRACTED_DIR,
                                  DATA_AUG_EXTRACTED_DIR)
    if not os.path.exists(aug_target_dir):
      os.rename(os.path.join(data_dir, DATA_AUG_EXTRACTED_DIR), aug_target_dir)

    train_writer = tf.python_io.TFRecordWriter(
      os.path.join(out_dir, train_record))
    for data in get_label_map_dict(os.path.join(data_dir, DATA_EXTRACTED_DIR),
                                   os.path.join('dataset', 'voc_lists',
                                                'train.txt')):
      example = dict_to_example(data)
      train_writer.write(example.SerializeToString())
    train_writer.close()

    val_writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, val_record))
    for data in get_label_map_dict(
        data_dir, os.path.join('dataset', 'voc_lists', 'val.txt')):
      example = dict_to_example(data)
      val_writer.write(example.SerializeToString())
      val_writer.close()


def get_pascal_palette():
  import numpy as np
  pallette = np.array(
    [
      [0, 0, 0],  # 0=background
      [128, 0, 0],  # 1=aeroplane
      [0, 128, 0],  # 2=bicycle
      [128, 128, 0],  # 3=bird
      [0, 0, 128],  # 4=boat
      [128, 0, 128],  # 5=bottle
      [0, 128, 128],  # 6=bus
      [128, 128, 128],  # 7=car
      [64, 0, 0],  # 8=cat
      [192, 0, 0],  # 9=chair
      [64, 128, 0],  # 10=cow
      [192, 128, 0],  # 11=diningtable
      [64, 0, 128],  # 12=dog
      [192, 0, 128],  # 13=horse
      [64, 128, 128],  # 14=motorbike
      [192, 128, 128],  # 15=person
      [0, 64, 0],  # 16=potted plant
      [128, 64, 0],  # 17=sheep
      [0, 192, 0],  # 18=sofa
      [128, 192, 0],  # 19=train
      [0, 64, 128]  # 20=tv/monitor
    ],
    dtype=np.int32)
  pallette = np.pad(
    pallette, ((0, 256 - pallette.shape[0]), (0, 0)),
    mode='constant',
    constant_values=0)
  pallette[255] = [224, 224, 192]  # 255=Ignorelabel
  return tf.constant(pallette)


def get_gt_img(logits_argmax, palette, num_images=1):
  if len(logits_argmax.shape) != 3:
    raise ValueError(
      'logits argmax should be a tensor of rank 3 with the shape [batch_size, height, width].'
    )
  n, h, w = logits_argmax.shape.as_list()
  if n < num_images:
    raise ValueError(
      'Batch size %d should be greater or equal than number of images to save %d.' \
      % (n, num_images))

  outputs = tf.gather_nd(
    params=tf.reshape(palette, [-1, 3]),
    indices=tf.reshape(logits_argmax, [n, -1, 1]))
  outputs = tf.cast(tf.reshape(outputs, [n, h, w, 3]), tf.uint8)
  return outputs


def pascal_voc_input_fn(is_training,
                        num_epochs=1,
                        batch_size=1,
                        img_scale_factor=1,
                        label_size=None,
                        buffer_size=500,
                        record_dir=DEFAULT_RECORD_DIR,
                        data_dir=DEFAULT_DATA_DIR):
  prepare_pascal_voc(data_dir, record_dir)

  height, width, channels_img, channels_gt = (500, 500, 3, 1)
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
    print(image)
    print(gt)
    image, gt = random_rotate(image, gt)
    return image, gt

  data_set = data_set.map(lambda record: dataset_parser(record))
  data_set = data_set.shuffle(buffer_size=buffer_size)
  data_set = data_set.repeat(num_epochs)

  iterator = data_set.batch(batch_size).make_one_shot_iterator()
  images, labels = iterator.get_next()
  labels = tf.image.resize_nearest_neighbor(labels, label_size)
  images = tf.reshape(images, [batch_size, out_width, out_height, channels_img])

  labels = tf.reshape(labels,
                      [batch_size, label_size[0], label_size[1], channels_gt])

  tf.summary.image('img/original', inv_preprocess(images, IMG_MEAN),
                   max_outputs=6)
  tf.summary.image('img/ground_truth',
                   get_gt_img(tf.squeeze(labels, axis=3), get_pascal_palette()),
                   max_outputs=6)
  return images, labels

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from utils.data import maybe_download, dict_to_example, get_label_map_dict
from utils.tf_image_processing import preprocess, scale, inv_preprocess, \
  random_rotate, get_gt_img, map_ground_truth

DEFAULT_DATA_DIR = '/tmp/dibco'
DATA_EXTRACTED_DIR = 'dibco'


def get_dibco_meta_data(data_dir=DEFAULT_DATA_DIR):
  n_train = 1
  n_test = 1
  if os.path.exists(os.path.join(data_dir, 'num_train')):
    with open(os.path.join(data_dir, 'num_train')) as f:
      n_train = int(f.read())
  if os.path.exists(os.path.join(data_dir, 'num_test')):
    with open(os.path.join(data_dir, 'num_test')) as f:
      n_test = int(f.read())
  return {
    # 'url': 'https://www.dropbox.com/s/62pps8pi5jfqzxg/dibco.zip?dl=1',  # split=0.2
    'url': 'https://www.dropbox.com/s/m5rfjfocjgrhicn/dibco.zip?dl=1',  # split=0.1
    # 'url': 'https://www.dropbox.com/s/4z2m9ndscs37i03/dibco.zip?dl=1',  # split=0.05
    'img_mean': [196.48484802, 188.59724426, 170.53767395],
    'num_classes': 2,
    'num_img_train': n_train,
    'num_img_test': n_test,
    'img_channels': 3,
    'gt_channels': 1,
    'default_img_height': 250,
    'default_img_width': 250,
  }


def prepare_dibco(data_dir=DEFAULT_DATA_DIR,
                  out_dir=None,
                  force=False):
  """ Downloads and extracts dibco dataset and its annotation data. """
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  if out_dir is None:
    out_dir = data_dir
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  train_record = 'train.record'
  test_record = 'test.record'
  num_train = 'num_train'
  num_test = 'num_test'

  if not (os.path.exists(os.path.join(data_dir, train_record)) and
          os.path.exists(os.path.join(data_dir, test_record)) and
          os.path.exists(os.path.join(data_dir, num_train)) and
          os.path.exists(os.path.join(data_dir, num_test))) or force:
    maybe_download(get_dibco_meta_data(data_dir)['url'], data_dir, force=force)

    extracted_dir = os.path.join(data_dir, DATA_EXTRACTED_DIR)

    with open(os.path.join(extracted_dir, 'train.txt')) as i_f:
      with open(os.path.join(data_dir, num_train), mode='w') as o_f:
        o_f.write(str(len(i_f.readlines())))
    train_writer = tf.python_io.TFRecordWriter(
      os.path.join(out_dir, train_record))
    for data in get_label_map_dict(
        extracted_dir, os.path.join(extracted_dir, 'train.txt')):
      example = dict_to_example(data)
      train_writer.write(example.SerializeToString())
    train_writer.close()

    with open(os.path.join(extracted_dir, 'test.txt')) as i_f:
      with open(os.path.join(data_dir, num_test), mode='w') as o_f:
        o_f.write(str(len(i_f.readlines())))
    val_writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, test_record))
    for data in get_label_map_dict(
        extracted_dir, os.path.join(extracted_dir, 'test.txt')):
      example = dict_to_example(data)
      val_writer.write(example.SerializeToString())
    val_writer.close()
    print()

  return get_dibco_meta_data(data_dir)


def get_dibco_palette():
  return tf.constant(
    [
      [255],  # Background
      [0],    # Writing
    ],
    dtype=tf.int32)


def dibco_input_fn(is_training,
                   num_epochs=1,
                   batch_size=1,
                   img_size=250,
                   img_scale_factor=1,
                   buffer_size=200,
                   data_dir=DEFAULT_DATA_DIR):
  if isinstance(img_size, tuple):
    out_height, out_width = img_size
  else:
    out_height = out_width = img_size
  out_size = (out_height, out_width)
  channels_img, channels_gt = (3, 1)
  process_heigh = int(out_height / img_scale_factor)
  process_width = int(out_width / img_scale_factor)

  record = os.path.join(
    data_dir, 'train.record' if is_training else 'test.record')
  if not os.path.exists(record):
    raise ValueError('TFRecord not found: {}.'.format(record) + \
                     'Did you download it `using prepare_dibco()`?')
  data_set = tf.data.TFRecordDataset(record)

  mean = get_dibco_meta_data(data_dir)['img_mean']

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

    image, gt = preprocess(image, gt, (process_heigh, process_width), mean,
                           is_training)
    image = scale(image, out_size=out_size, method='BILINEAR')
    gt = scale(gt, out_size=out_size)
    gt = map_ground_truth(gt, get_dibco_palette())
    image, gt = random_rotate(image, gt)
    return image, gt

  data_set = data_set.map(lambda record: dataset_parser(record))
  data_set = data_set.shuffle(buffer_size=buffer_size)
  data_set = data_set.repeat(num_epochs)

  iterator = data_set.batch(batch_size).make_one_shot_iterator()
  images, labels = iterator.get_next()

  images = tf.reshape(images, [batch_size, out_width, out_height, channels_img])
  labels = tf.reshape(labels, [batch_size, out_width, out_height, channels_gt])

  if is_training:
    mode_str = 'train'
  else:
    mode_str = 'eval'

  tf.summary.image(name=mode_str + '/original',
                   tensor=inv_preprocess(images, mean),
                   max_outputs=6)
  tf.summary.image(name=mode_str + '/ground_truth',
                   tensor=get_gt_img(tf.squeeze(labels, axis=3), get_dibco_palette()),
                   max_outputs=6)
  return images, labels


if __name__ == '__main__':
  prepare_dibco(force=False)

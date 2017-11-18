from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import io
import PIL.Image

from six.moves import urllib

import tensorflow as tf

DATA_URL = \
  'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
DATA_EXTRACTED_DIR = os.path.join('VOCdevkit', 'VOC2012')
DATA_URL_AUG = \
  'https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1'
DATA_AUG_EXTRACTED_DIR = 'SegmentationClassAug'
DEFAULT_DATA_DIR = '/tmp/pascal_voc'
DEFAULT_RECORD_DIR = '/tmp/pascal_voc'
IMG_MEAN = tf.constant([104.00698793, 116.66876762, 122.67891434])


def maybe_download_pascal_voc(url, data_dir, force=False):
  if not os.path.exists(data_dir):
    os.makedev(data_dir)

  filename = url.split('/')[-1]
  # For Dropbox download
  if filename.endswith('?dl=1'):
    filename = filename[:-5]
  filepath = os.path.join(data_dir, filename)

  if not os.path.exists(filepath) or force:
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  path = ''
  if filename.endswith('.tar'):
    import tarfile
    tarfile.open(filepath).extractall(data_dir)
  elif filename.endswith('.zip'):
    from zipfile import ZipFile
    ZipFile(filepath, 'r').extractall(data_dir)


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def dict_to_example(data):
  img_path = data['data_dir'] + data['img_path']
  gt_path = data['data_dir'] + data['gt_path']
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

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': _bytes_feature(encoded_img),
    'ground_truth/encoded': _bytes_feature(encoded_gt)}))
  return example


def get_label_map_dict(data_dir, data_set):
  data_dir = os.path.join(data_dir, DATA_EXTRACTED_DIR)
  dict = {}
  dict['data_dir'] = data_dir

  with tf.gfile.GFile(data_set, 'r') as fid:
    while True:
      line = fid.readline()
      if not line:
        break
      img, gt = line.rstrip().split(' ')
      dict['img_path'] = os.path.join(data_dir, img)
      dict['gt_path'] = os.path.join(data_dir, gt)
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
    maybe_download_pascal_voc(DATA_URL, data_dir, force=force)
    maybe_download_pascal_voc(DATA_URL_AUG, data_dir, force=force)
    aug_target_dir = os.path.join(
      data_dir, DATA_EXTRACTED_DIR, DATA_AUG_EXTRACTED_DIR)
    if not os.path.exists(aug_target_dir):
      os.rename(os.path.join(data_dir, DATA_AUG_EXTRACTED_DIR), aug_target_dir)

    train_writer = tf.python_io.TFRecordWriter(
      os.path.join(out_dir, train_record))
    for data in get_label_map_dict(
        data_dir, os.path.join('dataset', 'voc_lists', 'train.txt')):
      example = dict_to_example(data)
      train_writer.write(example.SerializeToString())
    train_writer.close()

    val_writer = tf.python_io.TFRecordWriter(
      os.path.join(out_dir, val_record))
    for data in get_label_map_dict(
        data_dir, os.path.join('dataset', 'voc_lists', 'val.txt')):
      example = dict_to_example(data)
      val_writer.write(example.SerializeToString())
      val_writer.close()


def preprocess_images(image, ground_truth, height, width, scale_factor,
                      is_training):
  depth_i = image.shape.as_list()[2]
  depth_gt = ground_truth.shape.as_list()[2]

  out_height = int(scale_factor * height)
  out_width = int(scale_factor * width)

  if is_training:
    combined = tf.concat([image, ground_truth], axis=2)

    combined = tf.image.resize_image_with_crop_or_pad(
      combined, int(height * 1.2), int(width * 1.2))

    combined = tf.random_crop(combined, [height, width, depth_i + depth_gt])

    combined = tf.image.random_flip_left_right(combined)

    image = combined[:, :, :depth_i]
    ground_truth = combined[:, :, depth_i:]

  else:
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    ground_truth = tf.image.resize_image_with_crop_or_pad(
      ground_truth, height, width)

  image = tf.image.resize_images(image, (out_height, out_width))
  ground_truth = tf.image.resize_images(ground_truth, (out_height, out_width),
                                        tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  image = tf.cast(image, dtype=tf.float32)
  image_mean_free = image - IMG_MEAN
  r, g, b = tf.split(image_mean_free, axis=2, num_or_size_splits=3)
  image_mean_free = tf.concat(axis=2, values=[b, g, r])
  ground_truth = tf.cast(ground_truth, dtype=tf.int32)
  return image_mean_free, ground_truth, image


def map_ground_truth(ground_truth, palette):
  """
  :param ground_truth: Rank 3 or 4 tensor: [(batch_size,) height, width, depth]
  :param palette: Color palette with rank [num_
  :return: one hot label
  """
  has_batch_size = True
  if len(ground_truth.shape) == 3:
    has_batch_size = False
    ground_truth = tf.expand_dims(ground_truth, axis=0)
  ground_truth = tf.cast(ground_truth, tf.int32)
  n, h, w, c = ground_truth.shape.as_list()
  num_classes, c_p = palette.shape.as_list()
  if c != c_p:
    raise ValueError(
      'Ground truth channels (%ds) do not match palette channels (%ds)' % (
        c, c_p))
  equality = tf.equal(tf.reshape(ground_truth, [n, h, w, 1, c]),
                      tf.reshape(palette, [num_classes, c]))
  label = tf.cast(tf.reduce_all(equality, axis=-1), tf.int32)
  if not has_batch_size:
    label = tf.squeeze(label, axis=0)
  return label


def get_pascal_palette():
  import numpy as np
  pallette = np.array([
    [0, 0, 0],  # 0=background
    [128, 0, 0],  # 1=aeroplane
    [0, 128, 0],  # 2=bicycle
    [128, 128, 0],  # 3=bird
    [0, 0, 128],  # 4=boat
    [128, 0, 128],  # 5=bottle
    [0, 128, 128],  # 6=bus
    [128, 128, 128],  # 7=car
    [64, 0, 0],  # 8=cat
    [192, 0, 0],  # 9=chair#
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
    [0, 64, 128]],  # 20=tv/monitor
    dtype=np.int32)
  # TODO: Make this better (maybe in get_gt_img()):
  pallette = np.pad(pallette, ((0, 256 - pallette.shape[0]), (0, 0)),
                    mode='constant', constant_values=0)
  pallette[255] = [224, 224, 192]  # 255=Ignorelabel
  return tf.constant(pallette)


def get_gt_img(logits_argmax, palette, num_images=1):
  if len(logits_argmax.shape) != 3:
    raise ValueError(
      'logits argmax should be a tensor of rank 3 with the shape [batch_size, height, width].')
  n, h, w = logits_argmax.shape.as_list()
  if n < num_images:
    raise ValueError(
      'Batch size %d should be greater or equal than number of images to save %d.' \
      % (n, num_images))

    # logits_argmax[logits_argmax==255] = 21

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

  def get_filenames():
    if is_training:
      return os.path.join(record_dir, 'train.record')
    else:
      return os.path.join(record_dir, 'val.record')

  height, width, channels_img, channels_gt = (500, 500, 3, 1)
  out_height = int(img_scale_factor * height)
  out_width = int(img_scale_factor * width)
  if label_size == None:
    label_size = (out_height, out_width)

  file_names = get_filenames()
  data_set = tf.contrib.data.TFRecordDataset(file_names)

  def dataset_parser(record):
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'ground_truth/encoded': tf.FixedLenFeature((), tf.string,
                                                 default_value='')}
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.image.decode_jpeg(parsed['image/encoded'],
                                 channels=channels_img)

    gt = tf.image.decode_png(parsed['ground_truth/encoded'],
                             channels=channels_gt)

    return preprocess_images(image, gt, height, width, img_scale_factor,
                             is_training)

  data_set = data_set.map(lambda value: dataset_parser(value))
  data_set = data_set.shuffle(buffer_size=buffer_size)
  data_set = data_set.repeat(num_epochs)

  iterator = data_set.batch(batch_size).make_one_shot_iterator()
  images, labels, orig_images = iterator.get_next()
  labels = tf.image.resize_nearest_neighbor(labels, label_size)
  images = tf.reshape(images, [batch_size, out_width, out_height, channels_img])

  labels = tf.reshape(labels, [batch_size, label_size[0], label_size[1],
                               channels_gt])

  tf.summary.image('img/original', orig_images, max_outputs=6)
  tf.summary.image('img/ground_truth', get_gt_img(
    tf.squeeze(labels, axis=3), get_pascal_palette()), max_outputs=6)
  return images, labels


# Test
def test_get_gt_img_map_ground_truth(sess):
  print('Testing get_gt_img() and map_ground_truth() ...')

  import tensorflow as tf
  import numpy as np

  # We build a random logits tensor of the requested size
  n = 2  # batch size
  h = w = 50  # image height, width
  num_classes = 21
  np.random.seed(1234)
  logits = np.random.random_sample(
    (n, h, w, num_classes))  # [n, h, w, num_classes]
  logits_argmax = tf.constant(np.argmax(logits, axis=3),
                              name='argmax')  # [n, h, w]

  palette = get_pascal_palette()  # [num_classes, c], c = 3

  print(logits_argmax)
  print(palette)
  reconstructed_gt = get_gt_img(logits_argmax, palette)  # [n, h, w, c]
  labels = map_ground_truth(reconstructed_gt, palette)  # [n, h, w, num_classes]
  labels_argmax = tf.argmax(labels, axis=3)  # [n, h, w]
  assert_op = tf.assert_equal(logits_argmax, labels_argmax)

  sess.run(assert_op)
  print('OK!')


if __name__ == '__main__':
  sess = tf.InteractiveSession()
  test_get_gt_img_map_ground_truth(sess)

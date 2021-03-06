from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import io

from six.moves import urllib
from PIL import Image

import tensorflow as tf


def maybe_download(url, data_dir, force=False):
  if not os.path.exists(data_dir):
    os.makedev(data_dir)

  filename = url.split('/')[-1]
  # For Dropbox download
  if filename.endswith('?dl=1'):
    filename = filename[:-5]
  filepath = os.path.join(data_dir, filename)

  if not os.path.exists(filepath) or force:

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

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


def read_img(path):
  with tf.gfile.GFile(path, 'rb') as fid:
    encoded_img = fid.read()
  return encoded_img

def get_img_size(encoded_img):
  encoded_img_io = io.BytesIO(encoded_img)
  return list(Image.open(encoded_img_io).size)


def dict_to_example(data, img_fn=read_img, gt_fn=read_img):
  img_path = os.path.join(data['data_dir'], data['img_path'])
  gt_path = os.path.join(data['data_dir'], data['gt_path'])
  sys.stdout.write('\r>> Processing %s, %s' %
                   (data['img_path'], data['gt_path']))
  sys.stdout.flush()
  encoded_img = img_fn(img_path)
  encoded_gt = gt_fn(gt_path)
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
            'image/encoded': _bytes_feature(encoded_img),
            'image/size': _int64_feature(get_img_size(encoded_img)),
            'ground_truth/encoded': _bytes_feature(encoded_gt),
            'ground_truth/size': _int64_feature(get_img_size(encoded_gt)),
          }))
  return example


def get_label_map_dict(data_dir, file_list):
  dict = {}
  dict['data_dir'] = data_dir

  with tf.gfile.GFile(file_list, 'r') as fid:
    while True:
      line = fid.readline()
      if not line:
        break
      img, gt = line.rstrip().split(' ')
      img = img.lstrip(r'\/')
      gt = gt.lstrip(r'\/')
      dict['img_path'] = img
      dict['gt_path'] = gt
      yield dict
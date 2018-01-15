from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

from models.unet import unet_model_fn_gen as model_fn_generator
from utils.tf_image_processing import get_gt_img, preprocess
from utils.py_image_processing import PatchGenerator, save_img
from dataset.dibco import get_dibco_meta_data, \
  get_dibco_palette
from dataset.hisdb import get_hisdb_meta_data, \
  get_hisdb_palette

parser = argparse.ArgumentParser()

parser.add_argument('images', type=str, nargs='+',
                    help='Path to the images to process')

# Basic model parameters.
parser.add_argument('--out_dir', type=str, default='/tmp/out',
                    help='The path to the dataset directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/unet_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--unet_depth', type=int, default=3,
                    help='The size of the Unet model to use.')

parser.add_argument('--filter_size', type=int, default=3,
                    help='Convolution filter size.')

parser.add_argument('--batch_size', type=int, default=1,
                    help='The number of images per batch.')

parser.add_argument('--img_patch_size', type=int, default=0,
                    help='Input image size for the sliding window. ' + \
                         'If not provided use a dataset specific default.')

parser.add_argument('--scale_factor', type=float, default=1,
                    help='Input image scale factor between (0, 1].')

parser.add_argument('--dataset', type=str, default='dibco',
                    help='The dataset to train with. Possible datasets: ' + \
                         '`dibco`, `hisdb`.')

FLAGS = parser.parse_args()

if FLAGS.dataset == 'dibco':
  get_meta = get_dibco_meta_data
  get_palette = get_dibco_palette
elif FLAGS.dataset == 'hisdb':
  get_meta = get_hisdb_meta_data
  get_palette = get_hisdb_palette
else:
  raise ValueError('Dataset not supported: %s.' % FLAGS.dataset)


def main(unused_argv):
  # Prepare the dataset
  img_meta = get_meta()
  if FLAGS.img_patch_size > 0:
    height = width = FLAGS.img_patch_size
  else:
    height = img_meta['default_img_height']
    width = img_meta['default_img_width']
  height = int(height * FLAGS.scale_factor)
  width = int(width * FLAGS.scale_factor)
  depth = img_meta['img_channels']

  print('Predicting {} given images.'.format(len(FLAGS.images)))

  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  model_fn = model_fn_generator(
    unet_depth=FLAGS.unet_depth,
    num_classes=img_meta['num_classes'],
    input_shape=[height, width, depth],
    filter_size=FLAGS.filter_size,
    get_gt_fn=lambda predictions: get_gt_img(predictions, get_palette()),
    crf_post_processing=True,
    save_dir=FLAGS.model_dir)

  classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir)

  # Predict output of given images
  for img_path in FLAGS.images:
    img_name = os.path.basename(img_path)
    print('Evaluation image %s... ' % img_name, end='')

    p_gen = PatchGenerator(img_path, (height, width))

    # Allocate some image buffer
    res_img = np.empty(p_gen.get_img_shape()[:-1], dtype=np.uint8)

    def input_fn(patch_gen, img_mean):
      data_set = tf.data.Dataset.from_generator(
        generator=patch_gen.get_patches_gen,
        output_types=tf.float32,
        output_shapes=tf.TensorShape(list(patch_gen.get_patch_size()) +
                                     [patch_gen.get_img_shape()[-1]])
      )

      data_set = data_set.map(
        lambda patches: preprocess(
          patches, None, patch_gen.get_patch_size(), img_mean, False)[0])

      data_set.prefetch(5)
      data_set.batch(1)
      iterator = data_set.make_one_shot_iterator()
      return iterator.get_next()

    predictions = classifier.predict(lambda: input_fn(p_gen, img_meta['img_mean']))
    for i, pred in enumerate(predictions):
      p_meta = p_gen.get_patch_meta(i)
      distribution = pred['probabilities'][0, :p_meta.height, :p_meta.width, 0]
      distribution *= 255
      distribution = tf.cast(distribution, dtype=tf.uint8)
      res_img[p_meta.pos_h:p_meta.pos_h + p_meta.height,
              p_meta.pos_w:p_meta.pos_w + p_meta.width] = distribution

    path = os.path.join(FLAGS.out_dir,
                        os.path.splitext(img_name)[0] + '_prediction' + '.png')
    save_img(res_img, path)
    print('finished!')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

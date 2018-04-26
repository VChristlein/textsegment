from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from models.unet import unet_model_fn_gen as model_fn_generator
from utils.tf_image_processing import get_gt_img
from dataset.dibco import prepare_dibco, \
                          dibco_input_fn, \
                          get_dibco_palette
from dataset.hisdb import prepare_hisdb, \
                          hisdb_input_fn, \
                          get_hisdb_palette

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/unet_data',
                    help='The path to the dataset directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/unet_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--unet_depth', type=int, default=3,
                    help='The size of the Unet model to use.')

parser.add_argument('--filter_size', type=int, default=3,
                    help='Convolution filter size.')

parser.add_argument('--train_epochs', type=int, default=1500,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=100,
                    help='The number of batches to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=1,
                    help='The number of images per batch.')

parser.add_argument('--buffer_size', type=int, default=0,
                    help='The number of images to buffer for training.')

parser.add_argument('--img_patch_size', type=int, default=0,
                    help='Input image size using random crop and pad. ' + \
                         'If not provided use a dataset specific default.')

parser.add_argument('--scale_factor', type=float, default=1,
                    help='Input image scale factor between (0, 1].')

parser.add_argument('--dataset', type=str, default='dibco',
                    help='The dataset to train with. Possible datasets: ' + \
                         '`dibco`, `hisdb`.')

parser.add_argument('--crf_training', type=bool, default=True,
                    help='After normal training train a downstream crf')

parser.add_argument('--only_crf', type=bool, default=False,
                    help='Start immediately with crf training')

parser.add_argument('--transfer', type=bool, default=False,
                    help='Use with pre-trained checkpoint file;' + \
                         'Only the downscale conv layers will get restored.')

FLAGS = parser.parse_args()

if FLAGS.dataset == 'dibco':
  prepare_dataset = prepare_dibco
  input_fn = dibco_input_fn
  get_palette = get_dibco_palette
elif FLAGS.dataset == 'hisdb':
  prepare_dataset = prepare_hisdb
  input_fn = hisdb_input_fn
  get_palette = get_hisdb_palette
else:
  raise ValueError('Dataset not supported: %s.' % FLAGS.dataset)

if FLAGS.buffer_size == 0:
  FLAGS.buffer_size = FLAGS.batch_size * 100

# Scale the learning rate linearly with the batch size.
_INITIAL_LEARNING_RATE = 0.1 * FLAGS.batch_size / 64
_NUM_EPOCHS_PER_DECAY = FLAGS.train_epochs / 2
_MOMENTUM = 0.9


def main(unused_argv):
  # Prepare the dataset
  meta_data = prepare_dataset(data_dir=FLAGS.data_dir)
  if FLAGS.img_patch_size > 0:
    height = width = FLAGS.img_patch_size
  else:
    height = meta_data['default_img_height']
    width = meta_data['default_img_width']
  height = int(height * FLAGS.scale_factor)
  width = int(width * FLAGS.scale_factor)
  depth = meta_data['img_channels']

  weight_decay = 2e-5 / (2 * meta_data['num_img_train'])
  batches_per_epoch = meta_data['num_img_train'] // FLAGS.batch_size

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1000)

  tf.logging.info('**********************************')
  tf.logging.info('Start training with dataset {}.'.format(FLAGS.dataset))
  tf.logging.info('**********************************')
  for use_crf in range(2 if FLAGS.crf_training else 1):
    decay_steps = int(batches_per_epoch * _NUM_EPOCHS_PER_DECAY)
    if FLAGS.only_crf:
      use_crf = True
    if not use_crf:
      if FLAGS.crf_training:
        # We now pretrain the model without the crf
        # Therefor don't use weight decay
        decay_steps = int(batches_per_epoch * _NUM_EPOCHS_PER_DECAY 
                            * FLAGS.train_epochs)
    else:
      tf.logging.info('Now training with a downstream CRF!')
      # The implementation only supports a batch size of 1
      FLAGS.batch_size = 1

    input_train = lambda: input_fn(
      is_training=True,
      img_size=(height, width),
      img_scale_factor=FLAGS.scale_factor,
      num_epochs=FLAGS.epochs_per_eval,
      batch_size=FLAGS.batch_size,
      buffer_size=FLAGS.buffer_size,
      data_dir=FLAGS.data_dir)

    input_val = lambda: input_fn(
      is_training=False,
      img_size=(height, width),
      img_scale_factor=FLAGS.scale_factor,
      data_dir=FLAGS.data_dir)

    model_fn = model_fn_generator(
      unet_depth=FLAGS.unet_depth,
      num_classes=meta_data['num_classes'],
      input_shape=[height, width, depth],
      filter_size=FLAGS.filter_size,
      get_gt_fn=lambda predictions: get_gt_img(predictions, get_palette()),
      initial_learning_rate=_INITIAL_LEARNING_RATE,
      learning_rate_decay_every_n_steps=decay_steps,
      momentum=_MOMENTUM,
      weight_decay=weight_decay,
      crf_post_processing=use_crf,
      save_dir=FLAGS.model_dir)

    warm_start = None
    if FLAGS.transfer:
      warm_start = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAGS.model_dir,
        vars_to_warm_start=".*transfer*"
      )

    classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      warm_start_from=warm_start
    )

    for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
      tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
      }

      logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

      classifier.train(
        input_fn=input_train,
        hooks=[logging_hook],
        steps=FLAGS.epochs_per_eval * meta_data['num_img_train'])

      # Evaluate the model and print results
      tf.logging.info('Evaluating model for epoch {} ...' \
          .format(i * FLAGS.epochs_per_eval))
      eval_results = classifier.evaluate(
        input_fn=input_val)
      tf.logging.info(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

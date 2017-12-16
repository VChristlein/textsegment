from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from models.unet import unet_model_fn_gen as model_fn_generator
# from dataset.pascal_voc import pascal_voc_input_fn as input_fn, \
#                                get_gt_img as gt_fn, \
#                                get_pascal_palette as get_palette
from dataset.dibco import dibco_input_fn as input_fn, \
                          get_gt_img as gt_fn, \
                          get_dibco_palette as get_palette

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/unet_data',
                    help='The path to the dataset directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/unet_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--unet_depth', type=int, default=3,
                    help='The size of the Unet model to use.')

parser.add_argument('--train_epochs', type=int, default=1000,
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

parser.add_argument('--crf_training', type=bool, default=False,
                    help='After normal training train a downstream crf')

FLAGS = parser.parse_args()

if FLAGS.buffer_size == 0:
  FLAGS.buffer_size = FLAGS.batch_size * 100

_NUM_CLASSES = 2

_HEIGHT = int(250 * FLAGS.scale_factor)
_WIDTH = int(250 * FLAGS.scale_factor)
_DEPTH = 3
_NUM_IMAGES = {
  'train': 68,
  'validation': 18,
}

# Scale the learning rate linearly with the batch size. When the batch size is
_INITIAL_LEARNING_RATE = 0.1 * FLAGS.batch_size / 64
_NUM_EPOCHS_PER_DECAY = FLAGS.train_epochs / 4
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

  input_train = lambda: input_fn(
    is_training=True,
    img_scale_factor=FLAGS.scale_factor,
    num_epochs=FLAGS.epochs_per_eval,
    batch_size=FLAGS.batch_size,
    buffer_size=FLAGS.buffer_size,
    data_dir=FLAGS.data_dir)

  input_val = lambda: input_fn(
    is_training=False,
    img_scale_factor=FLAGS.scale_factor,
    data_dir=FLAGS.data_dir)

  decay_steps = int(batches_per_epoch * _NUM_EPOCHS_PER_DECAY)

  for use_crf in range(2):
    model_fn = model_fn_generator(
      unet_depth=FLAGS.unet_depth,
      num_classes=meta_data['num_classes'],
      get_gt_fn=lambda predictions: gt_fn(predictions, get_palette()),
      input_shape=[height, width, depth],
      initial_learning_rate=_INITIAL_LEARNING_RATE,
      learning_rate_decay_every_n_steps=decay_steps,
      momentum=_MOMENTUM,
      weight_decay=weight_decay,
      crf_post_processing=use_crf,
      save_dir=FLAGS.model_dir)

    classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config)

    for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
      tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
      }

      logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

      classifier.train(
        input_fn=input_train,
        hooks=[logging_hook])

      # Evaluate the model and print results
      print('Evaluating model for epoch {} ...'.format(i * FLAGS.epochs_per_eval))
      eval_results = classifier.evaluate(
        input_fn=input_val)
      print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

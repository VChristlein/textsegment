from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from models.unet import unet_model_fn_gen as model_fn_generator
from dataset.pascal_voc import pascal_voc_input_fn as input_fn
from dataset.pascal_voc import get_gt_img as gt_fn
from dataset.pascal_voc import get_pascal_palette as get_palette

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/unet_data',
                    help='The path to the CIFAR-10 dataset directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/unet_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--unet_depth', type=int, default=3,
                    help='The size of the ResNet model to use.')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of batches to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=1,
                    help='The number of images per batch.')

parser.add_argument('--buffer_size', type=int, default=0,
                    help='The number of images to buffer for training.')

FLAGS = parser.parse_args()

if FLAGS.buffer_size == 0:
  FLAGS.buffer_size = FLAGS.batch_size * 100

_NUM_CLASSES = 21

_HEIGHT = 500
_WIDTH = 500
_DEPTH = 3
_NUM_IMAGES = {
  'train': 1464,
  'validation': 1449,
}

# Scale the learning rate linearly with the batch size. When the batch size is
_INITIAL_LEARNING_RATE = 0.1 * FLAGS.batch_size / 64
_NUM_EPOCHS_PER_DECAY = 100.0
_MOMENTUM = 0.9

_WEIGHT_DECAY = 2e-5 / (2 * _NUM_IMAGES['train'])

_BATCHES_PER_EPOCH = _NUM_IMAGES['train'] // FLAGS.batch_size


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1000)

  decay_steps = int(_BATCHES_PER_EPOCH * _NUM_EPOCHS_PER_DECAY)
  model_fn = model_fn_generator(
    unet_depth=FLAGS.unet_depth,
    num_classes=_NUM_CLASSES,
    ignore_last_class=True,
    get_gt_fn=lambda predictions: gt_fn(predictions, get_palette()),
    input_shape=[_HEIGHT, _WIDTH, _DEPTH],
    initial_learning_rate=_INITIAL_LEARNING_RATE,
    learning_rate_decay_every_n_steps=decay_steps,
    momentum=_MOMENTUM,
    weight_decay=_WEIGHT_DECAY)

  classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=FLAGS.model_dir,
    config=run_config)

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
      'learning_rate': 'learning_rate',
      'cross_entropy': 'cross_entropy',
      'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=1000)
    summary_writer = tf.summary.FileWriter(
      FLAGS.model_dir, tf.get_default_graph())
    summary_hook = tf.train.SummarySaverHook(
      save_steps=1000,
      summary_writer=summary_writer,
      scaffold=tf.train.Scaffold())

    classifier.train(
      input_fn=lambda: input_fn(
        is_training=True, 
        num_epochs=FLAGS.epochs_per_eval,
        batch_size=FLAGS.batch_size,
        buffer_size=FLAGS.buffer_size,
        record_dir=FLAGS.data_dir, 
        data_dir=FLAGS.data_dir),
      hooks=[logging_hook, summary_hook])

    # Evaluate the model and print results
    print('Evaluating model ...')
    eval_results = classifier.evaluate(
      input_fn=lambda: input_fn(
        is_training=False,
        record_dir=FLAGS.data_dir, 
        data_dir=FLAGS.data_dir))
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

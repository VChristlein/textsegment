from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from models.unet import unet_model_fn_gen as model_fn_generator
from dataset.pascal_voc import pascal_voc_input_fn as input_fn

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

parser.add_argument('--batch_size', type=int, default=20,
                    help='The number of images per batch.')

FLAGS = parser.parse_args()

_HEIGHT = 500
_WIDTH = 500
_DEPTH = 3
_NUM_CLASSES = 21

_NUM_IMAGES = {
  'train': 1464,
  'validation': 1449,
}

# Scale the learning rate linearly with the batch size. When the batch size is
# 128, the learning rate should be 0.1.
_INITIAL_LEARNING_RATE = 0.1 * FLAGS.batch_size / 128
_MOMENTUM = 0.9

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4

_BATCHES_PER_EPOCH = _NUM_IMAGES['train'] // FLAGS.batch_size


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1000)

  model_fn = model_fn_generator(
    unet_depth=FLAGS.unet_depth,
    num_classes=_NUM_CLASSES,
    input_shape=[_HEIGHT, _WIDTH, _DEPTH],
    initial_learning_rate=_INITIAL_LEARNING_RATE,
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
      tensors=tensors_to_log, every_n_iter=100)

    classifier.train(
      input_fn=lambda: input_fn(
        is_training=True, num_epochs=FLAGS.epochs_per_eval,
        batch_size=FLAGS.batch_size, num_classes=_NUM_CLASSES,
        record_dir=FLAGS.data_dir, data_dir=FLAGS.data_dir),
      hooks=[logging_hook])

    # Evaluate the model and print results
    print('Evaluating model ...')
    eval_results = classifier.evaluate(
      input_fn=lambda: input_fn(
        is_training=False,
        record_dir=FLAGS.data_dir, data_dir=FLAGS.data_dir))
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

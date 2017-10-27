from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from models.unet import unet_gen_model_fn as model_fn_generator
from dataset.pascal_voc import pascal_voc_input_fn as input_fn

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/unet_data',
                    help='The path to the CIFAR-10 dataset directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/unet_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--unet_depth', type=int, default=3,
                    help='The size of the ResNet model to use.')

FLAGS = parser.parse_args()

_HEIGHT = 500
_WIDTH = 500
_DEPTH = 3
_NUM_CLASSES = 21

_NUM_IMAGES = {
  'train': 1464,
  'validation': 1449,
}

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1000)

  model_fn = model_fn_generator(
      unet_depth=FLAGS.unet_depth,
      num_classes=_NUM_CLASSES,
      input_shape=[_HEIGHT, _WIDTH, _DEPTH])

  classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir,
      config=run_config)

  # Evaluate the model and print results
  eval_results = classifier.evaluate(
      input_fn=lambda: input_fn(
          is_training=False,
          record_dir=FLAGS.data_dir, data_dir=FLAGS.data_dir))
  print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import metrics
import tensorflow as tf


def f1_score(labels, predictions, metrics_collections=None, name=None):
  with tf.variable_scope(name, 'f1_score', (labels, predictions)):
    precision, precision_update = tf.metrics.precision(labels, predictions)
    tf.summary.scalar('precision', precision_update)
    recall, recall_update = tf.metrics.recall(labels, predictions)
    tf.summary.scalar('recall', recall_update)
    def compute_f1_score(rc, pr, name):
      return tf.div(2 * rc * pr, rc + pr, name)
    f1 = compute_f1_score(precision, recall, 'value')
    f1_update = compute_f1_score(precision_update, recall_update, 'update_op')
    if metrics_collections:
      tf.add_to_collections(metrics_collections, f1)
    return f1, f1_update

def psnr(labels, predictions, metrics_collections=None, name=None):
  with tf.variable_scope(name, 'psnr', (labels, predictions)):
    mse, mse_update = tf.metrics.mean_squared_error(labels, predictions)
    def compute_psnr(mse, c, name):
      return tf.multiply(10.0, tf.log(c / mse) / tf.log(10.0), name)
    psnr = compute_psnr(mse, 1.0, 'value')
    psnr_update = compute_psnr(mse_update, 1.0, 'update_op')
    if metrics_collections:
      tf.add_to_collections(metrics_collections, psnr)
    return psnr, psnr_update


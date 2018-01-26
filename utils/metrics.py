from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import metrics

from utils.py_image_processing import cv_distanceTransform


def pseudo_f1_score(labels, predictions, metrics_collections=None, name=None):
  with variable_scope.variable_scope(
      name, 'pseudo_f1_score', (predictions, labels)):
    import tensorflow as tf
    _r = metrics.recall(labels, predictions)
    _p = metrics.precision(labels, predictions)
    tp = metrics.true_positives(labels, predictions)
    fp = metrics.false_positives(labels, predictions)
    fn = metrics.false_negatives(labels, predictions)
    tf.summary.scalar('native_recall', _r[0])
    tf.summary.scalar('native_recall_1', _r[1])
    tf.summary.scalar('native_precision', _p[0])
    tf.summary.scalar('native_precision_1', _p[1])
    tf.summary.scalar('tp', tp[0])
    tf.summary.scalar('fp', fp[0])
    tf.summary.scalar('fn', fn[0])
    tf.summary.scalar('tp_1', tp[1])
    tf.summary.scalar('fp_1', fp[1])
    tf.summary.scalar('fn_1', fn[1])

    labels_f = math_ops.cast(labels, dtypes.float32)

    # Precision
    labels_uint = array_ops.squeeze(math_ops.cast(labels, dtypes.uint8),
                                    axis=-1)
    dist_p = script_ops.py_func(cv_distanceTransform, [labels_uint],
                                dtypes.float32, name='cv_dist_w')
    dist_p.set_shape(labels_uint.shape)
    tf.summary.histogram("dist_p", dist_p)
    pw = array_ops.where(math_ops.greater(dist_p, 8.),
                         x=array_ops.ones_like(labels_uint, dtypes.float32),
                         y=array_ops.where(
                           math_ops.equal(dist_p, 0.),
                           x=array_ops.ones_like(labels_uint, dtypes.float32),
                           y=1 + 1 / dist_p))
    tf.summary.histogram("pw", pw)

    pw = array_ops.reshape(pw, labels_f.shape)
    TP_p = math_ops.reduce_sum(predictions * labels_f * pw)
    TP_FN_p = math_ops.reduce_sum(predictions * pw)
    precision = array_ops.where(
      math_ops.greater(TP_FN_p, 0),
      math_ops.div(TP_p, TP_FN_p),
      0,
      name)
    tf.summary.scalar('TP_p', TP_p)
    tf.summary.scalar('TP_FN_p', TP_FN_p)
    tf.summary.scalar('precision', precision)

    # Recall
    labels_uint_r = 255 * (1 - labels_uint)
    dist_r = script_ops.py_func(cv_distanceTransform, [labels_uint_r],
                                dtypes.float32, name='cv_dist_r')
    tf.summary.histogram("dist_r", dist_r)
    dist_r.set_shape(labels_uint_r.shape)
    rw = array_ops.where(math_ops.greater_equal(dist_r, 2.),
                         x=1 - 1 / dist_r,
                         y=array_ops.zeros_like(labels_uint_r, dtypes.float32))
    tf.summary.histogram("rw", rw)

    rw = array_ops.reshape(rw, labels_f.shape)
    TP_r = math_ops.reduce_sum(labels_f * predictions * rw)
    TP_FN_r = math_ops.reduce_sum(labels_f * rw)
    recall = array_ops.where(
      math_ops.greater(TP_FN_r, 0),
      math_ops.div(TP_r, TP_FN_r),
      0,
      name)
    tf.summary.scalar('TP_r', TP_r)
    tf.summary.scalar('TP_FN_r', TP_FN_r)
    tf.summary.scalar('recall', recall)

    pf1 = 2 * precision * recall / (precision + recall)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, pf1)

    return pf1

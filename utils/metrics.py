from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import metrics

from utils.py_image_processing import cv_distanceTransform


def pseudo_recall(labels, predictions, metrics_collections=None,
                  updates_collections=None, name=None):
  with variable_scope.variable_scope(
      name, 'pseudo_recall', (predictions, labels)):
    labels_uint = 255 * (1 - math_ops.cast(labels, dtypes.uint8))
    dist = script_ops.py_func(cv_distanceTransform, [labels_uint],
                              dtypes.float32)
    weights = array_ops.where(math_ops.greater_equal(dist, 2),
                              x=1 - 1 / dist,
                              y=array_ops.zeros_like(dist))
    return metrics.recall(labels, predictions, weights, metrics_collections,
                          updates_collections, name)


def pseudo_precision(labels, predictions, metrics_collections=None,
                     updates_collections=None, name=None):
  with variable_scope.variable_scope(
      name, 'pseudo_precision', (predictions, labels)):
    labels_uint = math_ops.cast(labels, dtypes.uint8)
    dist = script_ops.py_func(cv_distanceTransform, [labels_uint],
                              dtypes.float32)
    weights = array_ops.where(math_ops.greater(dist, 8),
                              x=1.,
                              y=array_ops.where(
                                math_ops.equal(dist, 0.), x=1., y=1 + 1 / dist))
    return metrics.precision(labels, predictions, weights, metrics_collections,
                             updates_collections, name)


def f1_score(labels, predictions, weights=None, metrics_collections=None,
             updates_collections=None, name=None):
  with variable_scope.variable_scope(
      name, 'f1_score', (predictions, labels)):
    precision, update_pr = metrics.precision(labels, predictions, weights,
                                             metrics_collections,
                                             updates_collections, name)
    recall, update_re = metrics.recall(labels, predictions, weights,
                                       metrics_collections,
                                       updates_collections, name)

    f1 = 2 * precision * recall / (precision + recall)
    if metrics_collections:
      ops.add_to_collections(metrics_collections, f1)

    # TODO: Is this correct?
    update_op = 2 * precision * recall / (precision + recall)
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return f1, update_op


def pseudo_f1_score(labels, predictions, metrics_collections=None,
                    updates_collections=None, name=None):
  with variable_scope.variable_scope(
      name, 'f1_score', (predictions, labels)):
    precision, update_pr = pseudo_precision(labels, predictions,
                                            metrics_collections,
                                            updates_collections, name)
    recall, update_re = pseudo_recall(labels, predictions, metrics_collections,
                                      updates_collections, name)

    print(precision, recall)
    pf1 = 2 * precision * recall / (precision + recall)
    if metrics_collections:
      ops.add_to_collections(metrics_collections, pf1)

    # TODO: Is this correct?
    update_op = 2 * precision * recall / (precision + recall)
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return pf1, update_op

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from dataset.pascal_voc import get_gt_img
from utils.layers import *


def unet_block(inputs, filters, keep_prob, process_fn, is_training,
               data_format):
  shortcut = batch_norm_relu(
      inputs=inputs, is_training=is_training, data_format=data_format)

  shortcut = conv2d_fixed_padding(
      inputs=shortcut, filters=filters, kernel_size=3, strides=1,
      use_bias=False, data_format=data_format)
  shortcut = dropout(
      inputs=shortcut, keep_prob=keep_prob, is_training=is_training)
  shortcut = batch_norm_relu(
      inputs=shortcut, is_training=is_training, data_format=data_format)

  shortcut = conv2d_fixed_padding(
      inputs=shortcut, filters=filters, kernel_size=3, strides=1,
      use_bias=False, data_format=data_format)
  shortcut = dropout(
      inputs=shortcut, keep_prob=keep_prob, is_training=is_training)
  shortcut = batch_norm_relu(
      inputs=shortcut, is_training=is_training, data_format=data_format)

  output = process_fn(inputs)
  return shortcut, output


def unet(inputs, blocks, num_classes, is_training, data_format=None):
  if data_format == 'channels_first':
    # Convert from channels_last (NHWC) to channels_first (NCHW). This
    # provides a large performance boost on GPU.
    net = tf.transpose(inputs, [0, 3, 1, 2])
  else:
    net = inputs

  def pool(inputs):
    return max_pooling2d(
        inputs=inputs, pool_size=2, strides=2, padding='SAME',
        data_format=data_format)

  def conv_t(inputs, filters, out_h, out_w=None):
    if out_w == None:
      out_w = out_h
    return conv2d_t(
        inputs=inputs, filters=filters, out_h=out_h, out_w=out_w, kernel_size=2,
        strides=2, data_format=data_format)

  def conv_relu_out(inputs, filters):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        use_bias=False, data_format=data_format)
    return batch_norm_relu(
        inputs=inputs, is_training=is_training, data_format=data_format)

  shortcuts = OrderedDict()

  # down sampling
  for i in range(blocks["size"]):
    shortcuts[i], net = unet_block(
        inputs=net,
        filters=blocks["filters"][i],
        keep_prob=blocks["keep_prob"][i],
        process_fn=pool,
        is_training=is_training,
        data_format=data_format)

  # up sampling
  for i in reversed(range(blocks["size"])):
    _, net = unet_block(
        inputs=net,
        filters=blocks["filters"][i] * 2,
        keep_prob=blocks["keep_prob"][i],
        process_fn=lambda inputs: conv_t(inputs, blocks["filters"][i],
                                         shortcuts[i].shape[2]),
        is_training=is_training,
        data_format=data_format)
    if data_format == 'channels_first':
      concat_axis = 1
    else:
      concat_axis = 3
      net = tf.concat([shortcuts[i], net], axis=concat_axis)

  # output block
  _, net = unet_block(
      inputs=net,
      filters=blocks["filters"][i],
      keep_prob=blocks["keep_prob"][i],
      process_fn=lambda inputs: conv_relu_out(inputs, num_classes),
      is_training=is_training,
      data_format=data_format)

  return net


def unet_gen_model_fn(unet_depth,
                      num_classes,
                      input_shape,
                      initial_learning_rate=0.1,
                      momentum=0.9,
                      learning_rate_decay_every_n_steps=None,
                      weight_decay=2e-4,
                      data_format=None):
  """Generate model function"""
  model_params = {
    2: {"size": 2, "filters": [64, 128], "keep_prob": [0.75, 0.75]},
    3: {"size": 3, "filters": [64, 128, 256], "keep_prob": [0.75, 0.75, 0.75]},
    4: {"size": 4, "filters": [64, 128, 256, 512],
        "keep_prob": [0.75, 0.75, 0.75, 0.75]}
  }

  if unet_depth not in model_params:
    raise ValueError('Not a valid unet size.', unet_size)

  params = model_params[unet_size]

  if data_format is None:
    data_format = (
      'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  img_height = input_shape[0]
  img_width = input_shape[1]
  img_depth = input_shape[2]

  def unet_model_fn(features, labels, mode):
    tf.summary.image('images', features, max_outputs=6)

    inputs = tf.reshape(features, [-1, img_height, img_width, img_depth])
    logits = unet(inputs=inputs, blocks=params, num_classes=num_classes,
                  is_training=mode == tf.estimator.ModeKeys.TRAIN,
                  data_format=data_format)

    predictions = {
      'classes': tf.argmax(logits,
                           axis=(1 if data_format == 'channels_first' else 3)),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    if data_format == 'channels_first':
      labels = tf.transpose(labels, [0, 3, 2, 1])
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()

      # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
      if learning_rate_decay_every_n_steps is not None:
        boundaries = [int(learning_rate_decay_every_n_steps) for epoch in
                      [100, 150, 200]]
        values = [initial_learning_rate * decay for decay in
                  [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)
      else:
        learning_rate = tf.constant(initial_learning_rate)

      # Create a tensor named learning_rate for logging purposes
      tf.identity(learning_rate, name='learning_rate')
      tf.summary.scalar('learning_rate', learning_rate)

      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=momentum)

      # Batch norm requires update ops to be added as a dependency to the train_op
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    else:
      train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    result = get_gt_img(tf.argmax(labels, axis=1))
    tf.summary.image('result', result, max_outputs=6)

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

  return unet_model_fn


# test
_HEIGHT = 500
_WIDTH = 500
_DEPTH = 3
_BATCH_SIZE = 20

unet_size = 4
model_params = {
  2: {"size": 2, "filters": [64, 128], "keep_prob": [0.75, 0.75]},
  3: {"size": 3, "filters": [64, 128, 256], "keep_prob": [0.75, 0.75, 0.75]},
  4: {"size": 4, "filters": [64, 128, 256, 512],
      "keep_prob": [0.75, 0.75, 0.75, 0.75]}
}
params = model_params[unet_size]
features = tf.placeholder(tf.float32, [_HEIGHT, _WIDTH, _DEPTH])
inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
# inputs = tf.placeholder(tf.float32, [_BATCH_SIZE, _HEIGHT, _WIDTH, _DEPTH])
net = unet(inputs, blocks=params, num_classes=21, is_training=True,
           data_format='channels_first')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from utils.layers import *
from dataset.pascal_voc import get_gt_img
from dataset.pascal_voc import get_pascal_palette


def block(inputs, filters, keep_prob, process_fn, is_training,
          data_format):
  # print('In', inputs)
  inputs = batch_norm_relu(
    inputs=inputs, is_training=is_training, data_format=data_format)
  # print('relu', inputs)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=1,
    use_bias=False, data_format=data_format)
  inputs = dropout(
    inputs=inputs, keep_prob=keep_prob, is_training=is_training)
  inputs = batch_norm_relu(
    inputs=inputs, is_training=is_training, data_format=data_format)
  # print('Conv1', inputs)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=1,
    use_bias=False, data_format=data_format)
  inputs = dropout(
    inputs=inputs, keep_prob=keep_prob, is_training=is_training)
  inputs = batch_norm_relu(
    inputs=inputs, is_training=is_training, data_format=data_format)
  # print('Conv2', inputs)

  shortcut = inputs
  output = process_fn(inputs)
  return shortcut, output


def net(inputs, blocks, num_classes, is_training, data_format=None):
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
    shortcuts[i], net = block(
      inputs=net,
      filters=blocks["filters"][i],
      keep_prob=blocks["keep_prob"],
      process_fn=pool,
      is_training=is_training,
      data_format=data_format)

  # # up sampling
  # for i in reversed(range(blocks["size"])):
  #   _, net = block(
  #     inputs=net,
  #     filters=blocks["filters"][i] * 2,
  #     keep_prob=blocks["keep_prob"],
  #     process_fn=lambda inputs: conv_t(inputs, blocks["filters"][i],
  #                                      shortcuts[i].shape[2]),
  #     is_training=is_training,
  #     data_format=data_format)
  #   if data_format == 'channels_first':
  #     concat_axis = 1
  #   else:
  #     concat_axis = 3
  #     net = tf.concat([shortcuts[i], net], axis=concat_axis)

  # output block
  _, net = block(
    inputs=net,
    filters=blocks["filters"][0],
    keep_prob=blocks["keep_prob"],
    process_fn=lambda inputs: conv_relu_out(inputs, num_classes),
    is_training=is_training,
    data_format=data_format)

  return net


def get_testing_model_fn(depth,
                         num_classes,
                         input_shape,
                         initial_learning_rate=0.1,
                         momentum=0.9,
                         learning_rate_decay_every_n_steps=None,
                         weight_decay=2e-4,
                         data_format=None):
  """Generate model function"""
  model_params = {
    2: {"size": 2, "filters": [64, 128], "keep_prob": 0.75},
    3: {"size": 3, "filters": [64, 128, 256], "keep_prob": 0.75},
    4: {"size": 4, "filters": [64, 128, 256, 512], "keep_prob": 0.75}
  }

  if depth not in model_params:
    raise ValueError('Not a valid unet size.', depth)

  params = model_params[depth]

  if data_format is None:
    data_format = (
      'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  img_height = input_shape[0]
  img_width = input_shape[1]
  img_depth = input_shape[2]

  def test_model_fn(features, labels, mode):
    inputs = tf.reshape(features, [-1, img_height, img_width, img_depth])
    logits = net(inputs=inputs, blocks=params, num_classes=num_classes,
                 is_training=mode == tf.estimator.ModeKeys.TRAIN,
                 data_format=data_format)

    if data_format == 'channels_first':
      logits = tf.transpose(logits, [0, 2, 3, 1])

    flat_labels = tf.reshape(labels, [-1])
    flat_logits = tf.reshape(logits, [-1, num_classes])

    # Ignore the last class (the ignore label = 255)
    indices = tf.squeeze(tf.where(tf.less_equal(
      flat_labels, num_classes - 1)), 1)
    flat_labels = tf.cast(tf.gather(flat_labels, indices), tf.int32)
    flat_logits = tf.gather(flat_logits, indices)

    logits_argmax = tf.argmax(logits, axis=3)
    tf.summary.histogram('logits', logits_argmax)

    predictions = {
      'classes': logits_argmax,
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    tf.summary.histogram('labels', flat_labels)

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=flat_labels,
      logits=flat_logits)
    cross_entropy = tf.reduce_mean(cross_entropy)

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
        learning_rate = tf.train.exponential_decay(
          initial_learning_rate,
          global_step,
          learning_rate_decay_every_n_steps,
          0.1,
          staircase=True)
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

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    metrics = {'accuracy': accuracy}

    result = get_gt_img(predictions['classes'], get_pascal_palette())
    tf.summary.image('img/predicted_gt', result, max_outputs=6)

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

  return test_model_fn


if __name__ == '__main__':
  inputs = tf.placeholder(tf.float32, [1, 500, 500, 3])
  labels = tf.placeholder(tf.float32, [1, 63, 63, 21 + 1])
  model_fn = get_testing_model_fn(3, 21, [500, 500, 3])
  model_fn(inputs, labels, tf.estimator.ModeKeys.TRAIN)

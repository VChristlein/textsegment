from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf

from utils import layers, metrics as m


def unet_block(inputs, filters, filter_size, keep_prob, process_fn, is_training,
               data_format):
  inputs = layers.batch_norm_relu(
    inputs=inputs, is_training=is_training, data_format=data_format)

  inputs = layers.conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=filter_size, strides=1,
    use_bias=False, data_format=data_format)
  inputs = layers.dropout(
    inputs=inputs, keep_prob=keep_prob, is_training=is_training)
  inputs = layers.batch_norm_relu(
    inputs=inputs, is_training=is_training, data_format=data_format)

  inputs = layers.conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=filter_size, strides=1,
    use_bias=False, data_format=data_format)
  inputs = layers.dropout(
    inputs=inputs, keep_prob=keep_prob, is_training=is_training)
  inputs = layers.batch_norm_relu(
    inputs=inputs, is_training=is_training, data_format=data_format)

  shortcut = inputs
  output = process_fn(inputs)
  return shortcut, output


def unet(inputs, blocks, num_classes, filter_size, is_training,
         data_format=None):
  if data_format == 'channels_first':
    # Convert from channels_last (NHWC) to channels_first (NCHW). This
    # provides a large performance boost on GPU.
    net = tf.transpose(inputs, [0, 3, 1, 2])
  else:
    net = inputs

  def pool(inputs):
    return layers.max_pooling2d(
      inputs=inputs, pool_size=2, strides=2, padding='SAME',
      data_format=data_format)

  def conv_t(inputs, filters, out_h, out_w=None):
    if out_w == None:
      out_w = out_h
    return layers.conv2d_t(
      inputs=inputs, filters=filters, out_h=out_h, out_w=out_w, kernel_size=2,
      strides=2, data_format=data_format)

  def conv_relu_out(inputs, filters):
    inputs = layers.conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      use_bias=False, data_format=data_format)
    return layers.batch_norm_relu(
      inputs=inputs, is_training=is_training, data_format=data_format)

  shortcuts = OrderedDict()

  # down sampling
  for i in range(blocks["size"]):
    shortcuts[i], net = unet_block(
      inputs=net,
      filters=blocks["filters"][i],
      filter_size=filter_size,
      keep_prob=blocks["keep_prob"],
      process_fn=pool,
      is_training=is_training,
      data_format=data_format)

  # up sampling
  for i in reversed(range(blocks["size"])):
    _, net = unet_block(
      inputs=net,
      filters=blocks["filters"][i] * 2,
      filter_size=filter_size,
      keep_prob=blocks["keep_prob"],
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
    filter_size=filter_size,
    keep_prob=blocks["keep_prob"],
    process_fn=lambda inputs: conv_relu_out(inputs, num_classes),
    is_training=is_training,
    data_format=data_format)

  return net


def unet_model_fn_gen(unet_depth,
                      num_classes,
                      input_shape,
                      filter_size=3,
                      ignore_last_class=False,
                      get_gt_fn=None,
                      initial_learning_rate=0.1,
                      momentum=0.9,
                      learning_rate_decay_every_n_steps=None,
                      weight_decay=2e-4,
                      crf_post_processing=False,
                      save_dir=None,
                      data_format=None):
  """Generate model function"""
  model_params = {
    2: {"size": 2, "filters": [64, 128], "keep_prob": 0.75},
    3: {"size": 3, "filters": [64, 128, 256], "keep_prob": 0.75},
    4: {"size": 4, "filters": [64, 128, 256, 512], "keep_prob": 0.75},
    5: {"size": 5, "filters": [64, 128, 256, 256, 512], "keep_prob": 0.75}
  }

  if unet_depth not in model_params:
    raise ValueError('Not a valid unet size.', unet_depth)

  if get_gt_fn is None:
    get_gt_fn = lambda input: input

  params = model_params[unet_depth]

  if data_format is None:
    data_format = (
      'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  img_height = input_shape[0]
  img_width = input_shape[1]
  img_depth = input_shape[2]

  def unet_model_fn(features, labels, mode):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    inputs = tf.reshape(features, [-1, img_height, img_width, img_depth])
    logits = unet(inputs=inputs, blocks=params, num_classes=num_classes,
                  filter_size=filter_size, is_training=is_training,
                  data_format=data_format)

    if is_training:
      mode_str = 'train'
    else:
      mode_str = 'eval'

    if crf_post_processing:
      if logits.get_shape().as_list()[0] != 1:
        raise ValueError('Batch size must be one for crf training.')
      num_iterations = 10
    else:
      num_iterations = 0

    # Save summary before crf post processing
    logits_ = logits
    if data_format == 'channels_first':
      logits_ = tf.transpose(logits_, [0, 2, 3, 1])
    res_before_crf = get_gt_fn(tf.argmax(logits_, axis=3))
    tf.summary.image(mode_str + '/prediction_before_crf',
      res_before_crf, max_outputs=6)

    logits = layers.crf(
      inputs=[logits, tf.transpose(inputs, [0, 3, 1, 2])],
      num_classes=num_classes,
      data_format=data_format,
      num_iterations=num_iterations)

    if data_format == 'channels_first':
      # TODO: Is there a better way to compute the loss without a transpose?
      # Transform nchw back to nhwc for loss calculation
      logits = tf.transpose(logits, [0, 2, 3, 1])

    logits_argmax = tf.argmax(logits, axis=3)

    predictions = {
      'classes': logits_argmax,
      'result': get_gt_fn(logits_argmax),  # For image summary
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    flat_labels = tf.reshape(labels, [-1])
    flat_logits = tf.reshape(logits, [-1, num_classes])

    # Ignore the last class (ignore/void label)
    if ignore_last_class:
      indices = tf.squeeze(tf.where(tf.less_equal(
        flat_labels, num_classes - 1)), 1)
      flat_labels = tf.cast(tf.gather(flat_labels, indices), tf.int32)
      flat_logits = tf.gather(flat_logits, indices)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=flat_labels,
      logits=flat_logits)
    cross_entropy = tf.reduce_mean(cross_entropy)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    crf_log_tensor = tf.convert_to_tensor(crf_post_processing, dtype=tf.uint8)
    tf.summary.scalar('CRF', crf_log_tensor)
    tf.summary.scalar('batch_size', tf.shape(logits)[0])

    # Add weight decay to the loss.
    loss = cross_entropy + weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if is_training:
      global_step = tf.train.get_or_create_global_step()

      # Multiply the learning rate by 0.95 at 100, 150, and 200 epochs.
      if learning_rate_decay_every_n_steps is not None:
        learning_rate = tf.train.exponential_decay(
          learning_rate=initial_learning_rate,
          global_step=global_step,
          decay_steps=learning_rate_decay_every_n_steps,
          decay_rate=0.95,
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
    #fmeasure = m.f1_score(labels, predictions['probabilities'][:, :, :, 1])
    pfmeasure = m.pseudo_f1_score(labels,
                                  predictions['probabilities'][:, :, :, 1])
    metrics = {'accuracy': accuracy,
               # 'f1_score': fmeasure,
               # 'pf1_score': pfmeasure
               }

    tf.summary.image(mode_str + '/prediction', predictions['result'],
                     max_outputs=6)

    if num_classes == 2:
      distribution = predictions['probabilities'][:,:,:,0] * 255
      distribution = tf.expand_dims(tf.cast(distribution, dtype=tf.uint8), -1)
      tf.summary.image(mode_str + '/prediction_distribution',
                       distribution, max_outputs=6)

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    # tf.identity(accuracy[1], name='train_f1_score')
    # tf.summary.scalar('train_f1_score', fmeasure[1])
    tf.identity(pfmeasure, name='train_pf1_score')
    tf.summary.scalar('train_pf1_score', pfmeasure)

    if is_training:
      # tf.Estimator handles summaries during training
      hooks = None
    else:
      # But not during evaluation :(
      summary_hook = tf.train.SummarySaverHook(
        save_steps=1,
        output_dir=save_dir,
        summary_op=tf.summary.merge_all()
      )
      hooks = [summary_hook]

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics,
      evaluation_hooks=hooks
    )

  return unet_model_fn

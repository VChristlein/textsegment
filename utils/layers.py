import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training)
  # scale=True, training=is_training, fused=True)  # tf 1.3
  return tf.nn.relu(inputs)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the dataset either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, use_bias,
                         data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=use_bias,
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
      # kernel_initializer=tf.variance_scaling_initializer(),  # tf 1.3
      data_format=data_format)


# def conv2d_t_fixed_padding(inputs, filters, kernel_size, strides, use_bias,
#                            data_format):
#   """Strided transposed 2-D convolution with explicit padding."""
#   # The padding is consistent and is based only on `kernel_size`, not on the
#   # dimensions of `inputs` (as opposed to using `tf.layers.conv2d_transpose`
#   # alone).
#   if strides > 1:
#     inputs = fixed_padding(inputs, kernel_size, data_format)
#
#   return tf.layers.conv2d_transpose(
#       inputs=inputs,
#       filters=filters,
#       kernel_size=kernel_size,
#       strides=strides,
#       padding=('SAME' if strides == 1 else 'VALID'),
#       use_bias=use_bias,
#       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
#       # kernel_initializer=tf.variance_scaling_initializer(),  # tf 1.3
#       data_format=data_format)

def conv2d_t(inputs, filters, out_h, out_w, kernel_size=(2, 2), strides=(1, 1),
             padding='SAME', activation_fn=None, data_format='channels_last'):
  from tensorflow.python.ops import variable_scope as vs
  from tensorflow.python.ops import array_ops

  if isinstance(kernel_size, (int)):
    kernel_size = (kernel_size, kernel_size)
  if isinstance(strides, (int)):
    strides = (strides, strides)

  if data_format == 'channels_first':
    data_format_ = 'NCHW'
  else:  # 'channels_last':
    data_format_ = 'NHWC'
  inputs_shape = inputs.shape
  if data_format == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
  input_dim = inputs_shape[channel_axis]
  kernel_shape = kernel_size + (filters, input_dim)
  with vs.variable_scope(
      None, 'Conv2d_transpose', [inputs]) as sc:
    kernel = vs.get_variable(
        name='kernel',
        shape=kernel_shape,
        initializer=tf.contrib.layers.variance_scaling_initializer(),
        trainable=True,
        dtype=inputs.dtype)

  batch_size = inputs_shape[0]
  if data_format == 'channels_first':
    output_shape = (batch_size, filters, out_h, out_w)
    strides = (1, 1, strides[0], strides[1])
  else:
    output_shape = (batch_size, out_h, out_w, filters)
    strides = (1, strides[0], strides[1], 1)
  output_shape_tensor = array_ops.stack(output_shape)
  return tf.nn.conv2d_transpose(
      value=inputs, filter=kernel, output_shape=output_shape_tensor,
      strides=strides, padding=padding, data_format=data_format_)


def max_pooling2d(inputs, pool_size, strides, padding, data_format):
  return tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format
  )


def dropout(inputs, keep_prob, is_training):
  if keep_prob < 1:
    return tf.layers.dropout(inputs, 1 - keep_prob, training=is_training)
  else:
    return inputs

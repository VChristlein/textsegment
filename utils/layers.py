import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops

import utils.high_dim_filter_grad
custom_module = tf.load_op_library('./utils/high_dim_filter.so')

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
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
    kernel_initializer=init_ops.variance_scaling_initializer(),
      data_format=data_format)


def conv2d_t(inputs, filters, out_h, out_w, kernel_size=(2, 2), strides=(1, 1),
             padding='SAME', activation_fn=None, data_format='channels_last'):
  if isinstance(kernel_size, (int)):
    kernel_size = (kernel_size, kernel_size)
  if isinstance(strides, (int)):
    strides = (strides, strides)

  if data_format == 'channels_first':
    data_format_ = 'NCHW'
    channel_axis = 1
  else:  # 'channels_last':
    data_format_ = 'NHWC'
    channel_axis = -1

  inputs_shape = inputs.shape
  input_dim = inputs_shape[channel_axis]

  kernel_shape = kernel_size + (filters, input_dim)

  with vs.variable_scope(
      None, 'Conv2d_transpose', [inputs]) as sc:
    kernel = vs.get_variable(
        name='kernel',
        shape=kernel_shape,
        initializer=init_ops.variance_scaling_initializer(),
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


class CRF(base.Layer):
  def __init__(self,
               num_classes,
               theta_alpha=10.,
               theta_beta=10.,
               theta_gamma=10.,
               num_iterations=1,
               data_format='channels_last',
               trainable=True,
               name=None,
               **kwargs):
   super(CRF, self).__init__(trainable=trainable, name=name, **kwargs)

   self.num_classes = num_classes
   self.theta_alpha = theta_alpha
   self.theta_beta = theta_beta
   self.theta_gamma = theta_gamma
   self.num_iterations = num_iterations
   self.data_format = data_format
   self.initializer = init_ops.random_uniform_initializer
  # self.input_spec = base.InputSpec(ndim=3)


  def build(self, input_shape):
  #  input_shape = tensor_shape.TensorShape(input_shape)
    kernel_shape = (self.num_classes, self.num_classes)

    self.spatial_ker_weights = self.add_variable(
      name='spatial_ker_weights',
      shape=kernel_shape ,
      initializer=self.initializer(),
      dtype=self.dtype,
      trainable=True)

    self.bilateral_ker_weights = self.add_variable(
      name='bilateral_ker_weights',
      shape=kernel_shape,
      initializer=self.initializer(),
      dtype=self.dtype,
      trainable=True)

    self.compatibility_matrix = self.add_variable(
       name='compatibility_matrix',
       shape=kernel_shape,
       initializer=self.initializer(),
       dtype=self.dtype,
       trainable=True)

    def crf_op(inputs):
      # TODO: Implement batch version
      unaries = tf.squeeze(inputs[0], axis=0)
      if self.data_format != 'channels_first':
        unaries = tf.transpose(unaries, [2, 0, 1])
      rgb = tf.squeeze(inputs[1], axis=0)

      c, h, w = unaries.shape.as_list()
      all_ones = tf.ones((c, h, w), dtype=tf.float32)

      # Prepare filter normalization coefficients
      spatial_norm_vals = custom_module.high_dim_filter(
        all_ones, rgb, bilateral=False, theta_gamma=self.theta_gamma)
      bilateral_norm_vals = custom_module.high_dim_filter(
        all_ones, rgb, bilateral=True, theta_alpha=self.theta_alpha,
        theta_beta=self.theta_beta)

      q_values = unaries

      for i in range(self.num_iterations):
        softmax_out = tf.nn.softmax(q_values, dim=0)

        # Spatial filtering
        spatial_out = custom_module.high_dim_filter(
          softmax_out, rgb, bilateral=False, theta_gamma=self.theta_gamma)
        spatial_out = spatial_out / spatial_norm_vals

        # Bilateral filtering
        bilateral_out = custom_module.high_dim_filter(
          softmax_out, rgb, bilateral=True, theta_alpha=self.theta_alpha,
          theta_beta=self.theta_beta)
        bilateral_out = bilateral_out / bilateral_norm_vals

        # Weighting filter outputs
        message_passing = (
          tf.matmul(self.spatial_ker_weights, tf.reshape(spatial_out, (c, -1))) +
          tf.matmul(self.bilateral_ker_weights, tf.reshape(bilateral_out, (c, -1))))

        # Compatibility transform
        pairwise = tf.matmul(self.compatibility_matrix, message_passing)

        # Adding unary potentials
        pairwise = tf.reshape(pairwise, (c, h, w))
        q_values = unaries - pairwise

      q_values = tf.expand_dims(q_values, axis=0)
      if self.data_format == 'channels_first':
        return q_values
      else:
        return tf.transpose(q_values, [0, 2, 3, 1])


    self._crf_op = crf_op

    self.built = True


  def call(self, inputs):
    outputs = self._crf_op(inputs)

    return outputs


  def _compute_output_shape(self, input_shape):
    return input_shape[1]


def crf(inputs,
        num_classes,
        theta_alpha=10.,
        theta_beta=10.,
        theta_gamma=10.,
        num_iterations=1,
        data_format='channels_last',
        trainable=True,
        name=None):
  layer = CRF(num_classes=num_classes,
              theta_alpha=theta_alpha,
              theta_beta=theta_beta,
              theta_gamma=theta_gamma,
              num_iterations=num_iterations,
              data_format=data_format,
              trainable=trainable,
              name=name)
  return layer.apply(inputs)

import tensorflow as tf

"""Image processing tools for image segmentation."""


def get_random_angels(shape, deviation=0.5, distribution='NORMAL', name=None):
  """ Get random angels using a normal or a uniform distribution.

  `distribution` can be one of:
  *   `NORMAL`: Normal distribution
  *   `UNIFORM`: Uniform distribution

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output 
        tensor.
    deviation: A 0-D Tensor or Python value of type float32. For a normal 
        distribution deviation is used as its standard deviation. For a uniform
        distribution it is used as negativ minval and positiv maxval.
    distribution: A string.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random normal or uniform 
        values.
  """
  with tf.name_scope(name, 'GetRandomAngels', [shape]):
    if distribution == 'NORMAL':
      angels = tf.random_normal([shape], stddev=deviation)
    else:
      angels = tf.random_uniform([shape], minval=-1 * deviation,
                                 maxval=deviation)
    return angels


def random_rotate(images, ground_truth=None, stddev=0.5, name=None):
  """ Randomly rotate image(s).

  Use bilinear interpolation for `images` and nearest neighbor interpolation
  for `ground_truth`.

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW).
    ground_truth: A tensor like `images`. Can have a different number of
        channels.
    stddev: Standard deviation of a normal distribution. This will be used
       directly as angels in radiant. A standard deviation of 0.5 corresponds
       to a standard deviation of about 29 degrees.
    name: A name for the operation (optional).

  Returns:
    If ground_truth was not None a tuple else a single Tensor of dimension:
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  with tf.name_scope(name, 'RandomRotateImage', [images, ground_truth]):
    if images.get_shape().ndims == 4:
      shape = images.get_shape().as_list()[0]
    else:
      shape = 1
    angels = get_random_angels(shape, stddev)
    images = tf.contrib.image.rotate(images, angels, interpolation='BILINEAR')
    if ground_truth is None:
      return images
    ground_truth = tf.contrib.image.rotate(ground_truth, angels)
    return images, ground_truth


def rgb_to_bgr(images, name=None):
  """ Transforms a image tensor from RGB to BGR data format.

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), where num_channels
       must be 3.
    name: A name for the operation (optional).

  Returns:
    A Tensor of the same shape like `images`.
  """
  with tf.name_scope(name, 'RgbToBgr', [images]):
    axis = 2 if images.get_shape().ndims == 3 else 3
    r, g, b = tf.split(images, axis=axis, num_or_size_splits=3)
    images = tf.concat(axis=axis, values=[b, g, r])
    return images


def bgr_to_rgb(images, name=None):
  """ Transforms a image tensor from BGR to RGB data format.

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), where num_channels
       must be 3.
    name: A name for the operation (optional).

  Returns:
    A Tensor of the same shape like `images`.
  """
  with tf.name_scope(name, 'RgbToBgr', [images]):
    axis = 2 if images.get_shape().ndims == 3 else 3
    b, g, r = tf.split(images, axis=axis, num_or_size_splits=3)
    images = tf.concat(axis=axis, values=[r, g, b])
    return images


def preprocess(image, ground_truth, out_size, mean, is_training):
  """ Preprocess image and ground truth annotation.

  Args:
    image: 3-D Tensor of shape (num_rows, num_columns, num_channels) (HWC).
    ground_truth: 3-D Tensor of shape (num_rows, num_columns, num_channels)
        (HWC).
    out_size: Tuple of Int. Output height and width of the preprocessed image.
    mean: Python array or 1-D Tensor with the same length as the number of
        channels of `image`.
    is_training: Bool. If `True` the image and ground truth annotation will be
        augmented using shifting, left/right flipping and rotation.

    Returns:
      Tuple of 3-D Tensors: `image` and `ground_truth` tensors of shape
          (`height`, `width`, -1).
  """
  mean = tf.convert_to_tensor(mean, dtype=tf.float32)

  # TODO: Images and ground_truth should also have the possibility to be a
  #       4-D Tensor
  image = tf.convert_to_tensor(image)
  depth_i = image.shape.as_list()[2]

  if ground_truth is not None:
    ground_truth = tf.convert_to_tensor(ground_truth)
    depth_ground_truth = ground_truth.shape.as_list()[2]
  else:
    ground_truth = None
    depth_ground_truth = 0

  out_height, out_width = out_size

  if is_training:
    # Combine images so that we can randomly crop only one matrix for both
    # Image and ground truth label
    combined = tf.concat([image, ground_truth], axis=2)

    combined = tf.random_crop(
      combined, [out_height, out_width, depth_i + depth_ground_truth])

    combined = tf.image.random_flip_left_right(combined)

    image = combined[:, :, :depth_i]
    ground_truth = combined[:, :, depth_i:]

    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.random_brightness(image, max_delta=1.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

  else:
    image = tf.image.resize_image_with_crop_or_pad(image, out_height, out_width)
    if ground_truth is not None:
      ground_truth = tf.image.resize_image_with_crop_or_pad(
        ground_truth, out_height, out_width)

  image = image - mean
  image = rgb_to_bgr(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)

  if ground_truth is not None:
    ground_truth = tf.cast(ground_truth, dtype=tf.int32)
  return image, ground_truth


def inv_preprocess(images, mean, name=None):
  """ Transforms the tensor from BGR to RGB and adds the mean. """
  with tf.name_scope(name, "InvProprocessImages", [images, mean]):
    mean = tf.convert_to_tensor(mean, dtype=tf.float32)
    images = bgr_to_rgb(images)
    images = images + mean
    return images


def scale(images, out_size=None, scale_factor=1.0, method='NEAREST',
          name=None):
  """ Scale images to a given size.

  `method` can be one of:
  *   `NEAREST`: Nearest neighbor interpolation
  *   `BILINEAR`: Bilinear interpolation
  
  Args:
    images: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor
        of shape [height, width, channels].
    out_size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new
        size for the images.
    scale_factor: A 0-D Tensor or Python value of type float32. If out_size is 
        not given use the size of `images` and scale it by the given factor.
        Defaults to `1.0`.
    method: Resize method. Defaults do `NEAREST`
    name: A name for the operation (optional).

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`
  """
  with tf.name_scope(name, 'ScaleInput', [images]):
    if out_size is None:
      shape = images.get_shape().as_list()
      if images.get_shape().ndims == 3:
        out_size = [int(s * scale_factor) for s in shape[:2]]
      else:
        out_size = [int(s * scale_factor) for s in shape[1:3]]

    images = tf.image.resize_images(
      images, out_size,
      tf.image.ResizeMethod.NEAREST_NEIGHBOR if method is 'NEAREST' else \
        tf.image.ResizeMethod.BILINEAR)

    return images


def get_gt_img(logits_argmax, palette, num_images=1):
  if len(logits_argmax.shape) != 3:
    raise ValueError(
      'logits argmax should be a tensor of rank 3 with the shape [batch_size, height, width].'
    )
  n, h, w = logits_argmax.shape.as_list()
  if n < num_images:
    raise ValueError(
      'Batch size %d should be greater or equal than number of images to save %d.' \
      % (n, num_images))

  depth = palette.get_shape().as_list()[1]
  outputs = tf.gather_nd(
    params=palette,
    indices=tf.reshape(logits_argmax, [n, -1, 1]),
    name='GetGtImg')
  outputs = tf.cast(tf.reshape(outputs, [n, h, w, depth]), tf.uint8)
  return outputs


def map_ground_truth(ground_truth, palette, one_hot=True, name=None):
  """ Maps a ground truth image tensor to a label tensor.

  Args:
    ground_truth: Rank 3 or 4 tensor: [(batch_size,) height, width, depth].
    palette: Color palette with rank [num_classes, palette_depth].

  Returns:
    If `one_hot` is `True`, it returns a one hot label of shape
        [(batch_size,) height, width, 1], otherwise it will return a label of
        shape [(batch_size,) height, width, num_clases].

  Raises:
    ValueError: If the channels of ground truth and palette or not the same.

  """
  palette = tf.convert_to_tensor(palette)
  with tf.name_scope(name, 'MapGroundTruth', [ground_truth, palette]):
    is_batch = True
    if len(ground_truth.shape) == 3:
      is_batch = False
      ground_truth = tf.expand_dims(ground_truth, axis=0)
    ground_truth = tf.cast(ground_truth, tf.int32)
    n, h, w, c = ground_truth.shape.as_list()
    num_classes, c_p = palette.shape.as_list()
    if c != c_p:
      raise ValueError(
        'Ground truth channels (%ds) do not match palette channels (%ds)' %
        (c, c_p))
    equality = tf.equal(
      tf.reshape(ground_truth, [n, h, w, 1, c]),
      tf.reshape(palette, [num_classes, c]))
    label = tf.cast(tf.reduce_all(equality, axis=-1), tf.int32)
    if one_hot:
      label = tf.argmax(label, axis=3)
      label = tf.expand_dims(label, axis=-1)
    if not is_batch:
      label = tf.squeeze(label, axis=0)
    return label

import numpy as np
from PIL import Image
import cv2

LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2


def get_subwindows(im, tile_size, padding_size=32):
  height, width, = tile_size, tile_size
  y_stride, x_stride, = tile_size - (2 * padding_size), tile_size - (
      2 * padding_size)
  if (height > im.shape[0]) or (width > im.shape[1]):
    print("Invalid crop: crop dims larger than image %r" % im.shape)
    exit(1)
  ims = list()
  locations = list()
  y = 0
  y_done = False
  while y <= im.shape[0] and not y_done:
    x = 0
    if y + height > im.shape[0]:
      y = im.shape[0] - height
      y_done = True
    x_done = False
    while x <= im.shape[1] and not x_done:
      if x + width > im.shape[1]:
        x = im.shape[1] - width
        x_done = True
      locations.append((
        (y, x, y + height, x + width),
        (y + padding_size, x + padding_size, y + y_stride, x + x_stride),
        TOP_EDGE if y == 0 else (
          BOTTOM_EDGE if y == (im.shape[0] - height) else MIDDLE),
        LEFT_EDGE if x == 0 else (
          RIGHT_EDGE if x == (im.shape[1] - width) else MIDDLE)
      ))
      ims.append(im[y:y + height, x:x + width, :])
      x += x_stride
    y += y_stride

  return locations, ims


def stitch_together(locations, subwindows, size, tile_size, padding_size=32):
  output = np.zeros(size, dtype=np.float32)
  for location, subwindow in zip(locations, subwindows):
    subwindow = np.squeeze(subwindow)
    outer_bounding_box, inner_bounding_box, y_type, x_type = location
    y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1

    if y_type == TOP_EDGE:
      y_cut = 0
      y_paste = 0
      height_paste = tile_size - padding_size
    elif y_type == MIDDLE:
      y_cut = padding_size
      y_paste = inner_bounding_box[0]
      height_paste = tile_size - 2 * padding_size
    elif y_type == BOTTOM_EDGE:
      y_cut = padding_size
      y_paste = inner_bounding_box[0]
      height_paste = tile_size - padding_size

    if x_type == LEFT_EDGE:
      x_cut = 0
      x_paste = 0
      width_paste = tile_size - padding_size
    elif x_type == MIDDLE:
      x_cut = padding_size
      x_paste = inner_bounding_box[1]
      width_paste = tile_size - 2 * padding_size
    elif x_type == RIGHT_EDGE:
      x_cut = padding_size
      x_paste = inner_bounding_box[1]
      width_paste = tile_size - padding_size

    output[y_paste:y_paste + height_paste, x_paste:x_paste + width_paste] = \
      subwindow[y_cut:y_cut + height_paste, x_cut:x_cut + width_paste]

  return output


def open_img(path):
  return np.asarray(Image.open(path).convert('RGB'))


def save_img(image, path):
  img = Image.fromarray(image)
  if img.mode != 'RGB':
    img = img.convert('RGB')
  img.save(path, format='png')


def cv_distanceTransform(gt_label):
  # Iterate over all batches
  ret = np.empty_like(gt_label, dtype=np.float32)
  for i in range(gt_label.shape[0]):
    ret[i, :, :] = cv2.distanceTransform(gt_label[i, :, :], cv2.DIST_L2, 3)
  return ret

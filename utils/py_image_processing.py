import collections

import numpy as np
from PIL import Image

PatchMeta = collections.namedtuple('Patch', 'height, width, pos_h, pos_w')


class PatchGenerator():
  def __init__(self, img_path, patch_size):
    self.img = self._open_image(img_path)
    self.i_h = self.img.shape[0]  # image height
    self.i_w = self.img.shape[1]  # image width
    self.p_h = patch_size[0]  # patch height
    self.p_w = patch_size[1]  # patch width
    self.n_c = self.img.shape[-1]  # number of image channels
    self.n_p_h = int(
      np.ceil(self.i_h / self.p_h))  # number of patches for height
    self.n_p_w = int(
      np.ceil(self.i_w / self.p_w))  # number of patches for width
    self.patch_meta = []

  def _open_image(self, path):
    return np.asarray(Image.open(path))

  def get_patch_size(self):
    return self.p_h, self.p_w

  def get_img_shape(self):
    return self.i_h, self.i_w, self.n_c

  def get_patches_gen(self):
    for i in range(self.n_p_h):
      for j in range(self.n_p_w):
        patch = np.zeros((self.p_h, self.p_w, self.n_c), dtype=np.float32)
        to_h = min((i + 1) * self.p_h, self.i_h)
        to_w = min((j + 1) * self.p_w, self.i_w)
        p_h_ = self.p_h
        if i + 1 == self.n_p_h:
          modulo = to_h % self.p_h
          p_h_ = modulo if modulo > 0 else p_h_
        p_w_ = self.p_w
        if j + 1 == self.n_p_w:
          modulo = to_w % self.p_w
          p_w_ = modulo if modulo > 0 else p_w_
        self.patch_meta.append(PatchMeta(height=p_h_,
                                         width=p_w_,
                                         pos_h=i * self.p_h,
                                         pos_w=j * self.p_w))
        patch[:p_h_, :p_w_, :] = \
          self.img[i * self.p_h:to_h, j * self.p_w:to_w, :]
        yield patch

  def get_patch_meta(self, idx):
    return self.patch_meta[idx]


def save_img(image, path):
  img = Image.fromarray(image)
  img.save(path, format='png')

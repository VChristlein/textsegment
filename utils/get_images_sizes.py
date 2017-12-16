import os
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageStat

parser = argparse.ArgumentParser()

parser.add_argument('--directory', type=str)

parser.add_argument('--split', type=bool, default=True,
                    help='Do train/test split')

if __name__ == '__main__':
  FLAGS = parser.parse_args()

  directory = FLAGS.directory.replace('\'', '')
  height = width = list()
  mean = np.empty(shape=(3,), dtype=np.float32)
  filenames = []
  for filename in os.listdir(directory):
    if filename.endswith('.png') or filename.endswith('.jpg'):
      filenames.append(filename)
      img = Image.open(os.path.join(directory, filename))
      height_, width_ = img.size
      height.append(height_)
      width.append(width_)
      mean_ = np.array(ImageStat.Stat(img).mean, dtype=np.float32)
      if len(mean_) < 3:
        mean_ = np.append(mean_, [mean_, mean_])
      mean = np.vstack([mean, mean_])
  mean = mean[1:]

  if FLAGS.split:
    train, test = train_test_split(filenames, test_size=0.2, random_state=42)
  else:
    train, test = filenames, None

  train = np.sort(train)
  test = np.sort(test)

  with open('val.txt', 'w') as train_file:
    for file_name in train:
      train_file.write(os.path.join('images', file_name) + ' '
                       + os.path.join('gt', file_name) + '\n')

  if not FLAGS.split:
    with open('test.txt', 'w') as test_file:
      for file_name in test:
        test_file.write(os.path.join('images', file_name) + ' '
                        + os.path.join('gt', file_name) + '\n')

  print('Num images:', np.shape(mean)[0])
  print('Mean RGB:', np.mean(mean, axis=0))
  print('Num pixels:')
  print('max:  ', np.max(height), np.max(width))
  print('min:  ', np.min(height), np.min(width))
  print('mean: ', np.mean(height), np.mean(width))

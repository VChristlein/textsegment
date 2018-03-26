import os
import glob
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageStat

parser = argparse.ArgumentParser()

parser.add_argument('--directory', type=str)
parser.add_argument('--gt-directory', type=str, default=None)

parser.add_argument('--split', type=float, default=0.2,
                    help='Do train/test split.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for train test split.')

if __name__ == '__main__':
  FLAGS = parser.parse_args()

  directory = FLAGS.directory.replace('\'', '')
  if FLAGS.gt_directory is not None:
    gt_directory = FLAGS.gt_directory.replace('\'', '')
  else:
    gt_directory = directory
  height = width = list()
  mean = np.empty(shape=(3,), dtype=np.float32)
  filenames = []
  for filename in os.listdir(directory):
    if filename.endswith('.png') or filename.endswith('.jpg'):
      gt_filename = os.path.splitext(filename)[0]
      gt_filename = list(
        glob.glob(os.path.join(gt_directory, gt_filename + '*')))[0]
      gt_filename = os.path.basename(gt_filename)
      filenames.append((filename, gt_filename))
      img = Image.open(os.path.join(directory, filename))
      height_, width_ = img.size
      height.append(height_)
      width.append(width_)
      mean_ = np.array(ImageStat.Stat(img).mean, dtype=np.float32)
      if len(mean_) < 3:
        mean_ = np.append(mean_, [mean_, mean_])
      mean = np.vstack([mean, mean_])
  mean = mean[1:]

  if FLAGS.split > 0:
    print('Splitting into train and test set ({}%).'.format(FLAGS.split))
    train, test = train_test_split(filenames, test_size=FLAGS.split,
                                   random_state=FLAGS.seed)
  else:
    train, test = filenames, None

  train = np.sort(train, axis=0)
  test = np.sort(test, axis=0)

  with open('train.txt', 'w') as train_file:
    for file_name in train:
      train_file.write(os.path.join('images', file_name[0]) + ' '
                       + os.path.join('gt', file_name[1]) + '\n')

  if FLAGS.split > 0:
    with open('test.txt', 'w') as test_file:
      for file_name in test:
        test_file.write(os.path.join('images', file_name[0]) + ' '
                        + os.path.join('gt', file_name[1]) + '\n')

  print('Num images:', np.shape(mean)[0])
  print('Mean RGB:', np.mean(mean, axis=0))
  print('Num pixels:')
  print('max:  ', np.max(height), np.max(width))
  print('min:  ', np.min(height), np.min(width))
  print('mean: ', np.mean(height), np.mean(width))

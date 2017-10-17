import argparse
import os
import shutil

from zipfile import ZipFile
import rarfile

import tensorflow as tf


def download_and_extract(name, url, data_dir):
  tf.contrib.learn.datasets.base.maybe_download(name, data_dir,
                                                url + name)
  with ZipFile(os.path.join(data_dir, name)) as zip:
    zip.extractall(data_dir)

def get_dibco2016(data_dir):
  DIBCO2016_FILENAME = 'DIBCO2016_dataset-original.zip'
  DIBCO2016_EXTRACTNAME = 'DIPCO2016_dataset'
  DIBCO2016_FILENAME_GT = 'DIBCO2016_dataset-GT.zip'
  DIBCO2016_EXTRACTNAME_GT = 'DIPCO2016_Dataset_GT'
  DIBCO2016_URL = 'http://vc.ee.duth.gr/h-dibco2016/benchmark/'

  print('Download from {}{} and extract.'.format(DIBCO2016_URL,
                                                 DIBCO2016_FILENAME))
  download_and_extract(DIBCO2016_FILENAME, DIBCO2016_URL, data_dir)

  print('Download from {}{} and extract.'.format(DIBCO2016_URL,
                                                 DIBCO2016_FILENAME_GT))
  download_and_extract(DIBCO2016_FILENAME_GT, DIBCO2016_URL, data_dir)


def convert_to_tfrecord(input_files, output_file):
  pass

def main(data_dir):
  # file_names = _get_file_names()
  # input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
  # for mode, files in file_names.items():
  #   input_files = [os.path.join(input_dir, f) for f in files]
  #   output_file = os.path.join(data_dir, mode + '.tfrecords')
  #   try:
  #     os.remove(output_file)
  #   except OSError:
  #     pass
  #   # Convert to tf.train.Example and write the to TFRecords.
  #   convert_to_tfrecord(input_files, output_file)
  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data-dir',
    type=str,
    default='',
    help='Directory to download and extract the data to.')

  args = parser.parse_args()
  main(args.data_dir)

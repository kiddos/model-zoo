import numpy as np
from scipy.ndimage import imread
import os
import sqlite3
import logging
import sys
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger('face detection data util')
logger.setLevel(logging.INFO)


def compute_image_shape_mean(image_dir):
  if os.path.isdir(image_dir):
    logger.info('loading %s...' % (image_dir))
    directories = os.listdir(image_dir)
    image_count = 0
    width_mean = 0
    height_mean = 0
    try:
      for d in directories:
        folder = os.path.join(image_dir, d)
        image_files = os.listdir(folder)
        for img in image_files:
          image_count += 1
          image_path = os.path.join(folder, img)
          image = imread(image_path)
          width_mean += image.shape[1]
          height_mean += image.shape[0]
          sys.stdout.write('\rcount: %d, shape: %d, %d' %
            (image_count, image.shape[1], image.shape[0]))
          sys.stdout.flush()
          del image
    except KeyboardInterrupt:
      logger.warn('\r\ninterrupt')
  else:
    logger.info('unabel to find %s.' % (image_dir))
  width_mean /= float(image_count)
  height_mean /= float(image_count)
  logger.info('\r\nimage count: %d, image width mean: %f, height mean: %f' %
    (image_count, width_mean, height_mean))


def collect_image_path(image_dir):
  image_paths = []
  if os.path.isdir(image_dir):
    logger.info('loading %s...' % (image_dir))
    directories = os.listdir(image_dir)
    for d in directories:
      folder = os.path.join(image_dir, d)
      image_files = os.listdir(folder)
      for img in image_files:
        image_path = os.path.join(folder, img)
        image_paths.append(image_path)
  return image_paths


def main():
  parser = ArgumentParser()
  parser.add_argument('--train-images', dest='train_images',
    default='WIDER_train/images',
    help='directories containing the folders of images for training')
  parser.add_argument('--train-label', dest='train_labels',
    default='wider_face_split/wider_face_train.mat')
  parser.add_argument('--test-label', dest='test_labels',
    default='wider_face_split/wider_face_test.mat')
  parser.add_argument('--val-label', dest='val_labels',
    default='wider_face_split/wider_face_val.mat')
  args = parser.parse_args()

  compute_image_shape_mean(args.train_images)


if __name__ == '__main__':
  main()

import numpy as np
import logging
import os
import random
import math
from scipy.misc import imresize
from scipy.ndimage import imread
from scipy.io import loadmat
from argparse import ArgumentParser

from yolo_face_v9 import IMAGE_WIDTH, IMAGE_HEIGHT, GRID_SIZE
from face_detection_data_util import collect_image_path


logging.basicConfig()
logger = logging.getLogger('yolo face batcher')
logger.setLevel(logging.INFO)


class ImageBatch(object):
  def __init__(self, image_dir, label_txt):
    print(len(collect_image_path(image_dir)))
    self.image_paths = {path: [] for path in collect_image_path(image_dir)}
    if os.path.isfile(label_txt):
      logger.info('loading %s...' % (label_txt))
      image_count = 0
      with open(label_txt, 'r') as f:
        while True:
          line = f.readline()
          if not line:
            break
          image_count += 1
          path_name = os.path.join(image_dir, line)[:-1]
          num_box = int(f.readline())
          boxes = []
          for i in range(num_box):
            box = [int(e) for e in f.readline().split(' ')[:-1]]
            box[0] += box[2] / 2
            box[1] += box[3] / 2
            boxes.append(box[:4])
          self.image_paths[path_name] = boxes
        assert len(self.image_paths) == image_count
        logger.info('image count: %d' % (image_count))
    else:
      logger.warn('%s not found' % (label_txt))

  def next(self, path_name):
    image = imread(path_name)
    indicator = np.zeros(shape=[GRID_SIZE, GRID_SIZE, 1])
    coord = np.zeros(shape=[GRID_SIZE, GRID_SIZE, 2])
    size = np.zeros(shape=[GRID_SIZE, GRID_SIZE, 2])
    for box in self.image_paths[path_name]:
      x = np.minimum(np.maximum(
        int(box[0] * GRID_SIZE / image.shape[1]), 0), GRID_SIZE - 1)
      y = np.minimum(np.maximum(
        int(box[1] * GRID_SIZE / image.shape[0]), 0), GRID_SIZE - 1)
      indicator[y, x, 0] = 1.0

      coord[y, x, 0] = float(box[0]) / image.shape[1]
      coord[y, x, 1] = float(box[1]) / image.shape[0]
      size[y, x, 0] = float(box[2]) / image.shape[1]
      size[y, x, 1] = float(box[3]) / image.shape[0]
    return imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)), \
        indicator, coord, size

  def next_batch(self, batch_size):
    images = []
    indicators = []
    coords = []
    sizes = []
    for b in range(batch_size):
      path_name = random.choice(self.image_paths.keys())
      image, indicator, coord, size = self.next(path_name)
      images.append(image)
      indicators.append(indicator)
      coords.append(coord)
      sizes.append(size)
    return np.array(images, dtype=np.uint8), \
        np.array(indicators), np.array(coords), np.array(sizes)


def test():
  parser = ArgumentParser()
  parser.add_argument('--train-images', dest='train_images',
    default='WIDER_train/images',
    help='directories containing the folders of images for training')
  parser.add_argument('--train-label', dest='train_labels',
    default='wider_face_split/wider_face_train_bbx_gt.txt')
  parser.add_argument('--test-label', dest='test_labels',
    default='wider_face_split/wider_face_test_filelist.txt')
  parser.add_argument('--val-label', dest='val_labels',
    default='wider_face_split/wider_face_val_bbx_gt.txt')
  args = parser.parse_args()

  batch = ImageBatch(args.train_images, args.train_labels)
  for i in range(10):
    image_batch, indicator_batch, coord_batch, size_batch = \
        batch.next_batch(1)
    logger.info('image batch shape: %s' % (str(image_batch.shape)))
    logger.info('indicator batch shape: %s' % (str(indicator_batch.shape)))
    logger.info('coord batch shape: %s' % (str(coord_batch.shape)))
    logger.info('size batch shape: %s' % (str(size_batch.shape)))
    print(size_batch)


if __name__ == '__main__':
  test()

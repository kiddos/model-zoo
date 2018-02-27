import sqlite3
import numpy as np
import logging
import os
import unittest
import random
import math
from PIL import Image
from argparse import ArgumentParser

logging.basicConfig()
logger = logging.getLogger('plants')
logger.setLevel(logging.INFO)


class PlantLoader(object):
  def __init__(self, dbname, input_size, training_percent=0.8):
    self.percent = training_percent
    self.input_size = input_size

    if os.path.isfile(dbname):
      self.connection = sqlite3.connect(dbname)
      self.cursor = self.connection.cursor()

      self._setup()
    else:
      logger.warning('%s not found' % (dbname))

  def __del__(self):
    logger.info('closing connection...')
    if hasattr(self, 'connection'):
      self.connection.close()

  def _setup(self):
    self.cursor.execute("""SELECT * FROM meta;""")
    meta = self.cursor.fetchone()
    self.width = meta[0]
    self.height = meta[1]
    self.channel = meta[2]
    self.output_size = 12

  def load_data(self):
    if not hasattr(self, 'connection'):
      logger.warning('db not found')
      return

    logger.info('loading images and labels...')
    self.cursor.execute("""SELECT image, classes FROM plants;""")
    raw_data = self.cursor.fetchall()

    self.images = []
    self.labels = []
    for d in raw_data:
      img = np.frombuffer(d[0], np.uint8)
      self.images.append(np.reshape(img,
        [self.height, self.width, self.channel]))
      label = d[1] - 1
      self.labels.append([1 if label == l else 0
        for l in range(self.output_size)])
    self.images = np.array(self.images)
    self.labels = np.array(self.labels)

    self.cursor.execute("""SELECT className FROM classes;""")
    self.label_name = self.cursor.fetchall()

    data_size = len(self.images)
    index = np.random.permutation(np.arange(data_size))
    training_size = int(data_size * self.percent)
    training_index = index[:training_size]
    validation_index = index[training_size:]

    self.training_data = np.copy(self.images[training_index, :])
    self.training_label = np.copy(self.labels[training_index, :])
    self.validation_data = np.copy(self.images[validation_index, :])
    self.validation_label = np.copy(self.labels[validation_index, :])

    self.images = np.copy(self.images[index, :])
    self.labels = np.copy(self.labels[index, :])

    self.cursor.execute("""SELECT fileName, image FROM test;""")
    raw_test_data = self.cursor.fetchall()
    self.test_files = []
    self.test_images = []
    for d in raw_test_data:
      img = np.frombuffer(d[1], np.uint8)
      self.test_files.append(d[0])
      self.test_images.append(np.reshape(img,
        [self.height, self.width, self.channel]))
    self.test_images = np.array(self.test_images)

  def sample(self, batch_size, all=False):
    data = self.training_data
    labels = self.training_label
    if all:
      data = self.images
      labels = self.labels

    batch_data = []
    batch_label = []
    max_pad = int(self.input_size * 0.2)
    for i in range(batch_size):
      index = random.randint(0, len(data) - 1)
      img = Image.fromarray(data[index, ...])

      # rotate
      angle = random.randint(-180, 180)
      img = img.rotate(angle, resample=Image.BICUBIC)

      # crop out black
      orig_size = min(img.size)
      rad = np.abs(angle / 180.0 * math.pi)
      new_size = orig_size / (np.cos(rad) + np.sin(rad))
      pad = (orig_size - new_size) / 2
      img = img.crop([
        pad + random.randint(0, max_pad),
        pad + random.randint(0, max_pad),
        img.size[0] - pad - random.randint(0, max_pad),
        img.size[1] - pad - random.randint(0, max_pad)
      ])

      img = img.resize([self.input_size, self.input_size])
      batch_data.append(np.array(img, np.uint8))
      batch_label.append(labels[index])
    return np.array(batch_data), np.array(batch_label)

  def get_output_size(self):
    return self.output_size

  def get_width(self):
    return self.width

  def get_height(self):
    return self.height

  def get_data(self):
    return self.images

  def get_label(self):
    return self.labels

  def get_label_name(self):
    return self.label_name

  def get_training_data(self):
    return self.training_data

  def get_training_labels(self):
    return self.training_label

  def get_validation_data(self):
    data = []
    for i in range(len(self.validation_data)):
      img = Image.fromarray(self.validation_data[i, ...])
      img = img.resize([self.input_size, self.input_size])
      data.append(np.array(img, np.uint8))
    return np.array(data)

  def get_validation_labels(self):
    return self.validation_label

  def get_test_files(self):
    return self.test_files

  def get_test_images(self):
    return self.test_images


class TestPlantLoader(unittest.TestCase):
  def setUp(self):
    parser = ArgumentParser()
    parser.add_argument('--dbname', dest='dbname', default='plants.sqlite3',
      type=str, help='db to load')
    args = parser.parse_args()

    self.loader = PlantLoader(args.dbname, 64)
    self.loader.load_data()

    self.label_name = self.loader.get_label_name()
    self.label_name = [l[0] for l in self.label_name]

  def test_label_name(self):
    self.assertTrue(u'Scentless Mayweed' in self.label_name)
    self.assertTrue(u'Common Chickweed' in self.label_name)
    self.assertTrue(u'Small-flowered Cranesbill' in self.label_name)
    self.assertTrue(u'Black-grass' in self.label_name)
    self.assertTrue(u'Charlock' in self.label_name)
    self.assertTrue(u'Sugar beet' in self.label_name)
    self.assertTrue(u'Shepherds Purse' in self.label_name)
    self.assertTrue(u'Cleavers' in self.label_name)
    self.assertTrue(u'Loose Silky-bent' in self.label_name)
    self.assertTrue(u'Common wheat' in self.label_name)
    self.assertTrue(u'Maize' in self.label_name)
    self.assertTrue(u'Fat Hen' in self.label_name)

  def test_validation_set(self):
    train = self.loader.get_training_data(), self.loader.get_training_labels()
    logger.info('training data shape: %s', str(train[0].shape))
    logger.info('training label shape: %s', str(train[1].shape))

    valid = self.loader.validation_data, \
      self.loader.get_validation_labels()
    logger.info('validation data shape: %s', str(valid[0].shape))
    logger.info('validation label shape: %s', str(valid[1].shape))

    for i in range(len(valid)):
      eq = False
      for j in range(len(train)):
        eq |= (train[0][j, ...] == valid[0][i, ...]).all()
      self.assertFalse(eq)

  def test_sample(self):
    try:
      import cv2
    except:
      raise Exception

    display = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
    sample_data, sample_label = self.loader.sample(64)

    self.assertEqual(len(sample_data), len(sample_data))
    for i in range(len(sample_data)):
      r = i % 8
      c = i / 8
      display[(r * 64):((r + 1) * 64), (c * 64):((c + 1) * 64), :] = \
        sample_data[i, ...]
      l = sample_label[i].argmax()
      cv2.putText(display, self.label_name[l], (c * 64, r * 64 + 10),
        cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

    cv2.imshow('Samples', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

  def test_validation_data(self):
    try:
      import cv2
    except:
      raise Exception

    display = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
    valid_data = self.loader.get_validation_data()
    valid_label = self.loader.get_validation_labels()

    self.assertEqual(len(valid_data), len(valid_label))
    index = np.random.permutation(np.arange(len(valid_data)))[:64]
    valid_data = valid_data[index, ...]
    valid_label = valid_label[index, :]

    for i in range(len(valid_data)):
      r = i % 8
      c = i / 8
      display[(r * 64):((r + 1) * 64), (c * 64):((c + 1) * 64), :] = \
        valid_data[i, ...]
      l = valid_label[i].argmax()
      cv2.putText(display, self.label_name[l], (c * 64, r * 64 + 10),
        cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

    cv2.imshow('Validation data', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)



if __name__ == '__main__':
  unittest.main()

import sqlite3
import numpy as np
import logging
import os
from argparse import ArgumentParser

logging.basicConfig()
logger = logging.getLogger('plants')
logger.setLevel(logging.INFO)


class PlantLoader(object):
  def __init__(self, dbname, training_percent=0.9):
    self.percent = training_percent
    if os.path.isfile(dbname):
      self.connection = sqlite3.connect(dbname)
      self.cursor = self.connection.cursor()

      self._setup()
    else:
      logger.warning('%s not found' % (dbname))

  def __del__(self):
    logger.info('closing connection...')
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
    return self.validation_data

  def get_validation_labels(self):
    return self.validation_label

  def get_test_files(self):
    return self.test_files

  def get_test_images(self):
    return self.test_images

def main():
  parser = ArgumentParser()
  parser.add_argument('--dbname', dest='dbname', default='plants.sqlite3',
    type=str, help='db to load')
  args = parser.parse_args()

  p = PlantLoader(args.dbname)
  p.load_data()

  logger.info('all data shape: %s' % str(p.get_data().shape))
  logger.info('all label shape: %s' % str(p.get_label().shape))
  assert len(p.get_data()) == len(p.get_label())

  logger.info('label names:')
  for l in p.get_label_name():
    logger.info(l[0])
  assert len(p.get_label_name()) == p.get_output_size()

  logger.info('training data shape: %s' % str(p.get_training_data().shape))
  logger.info('training label shape: %s' % str(p.get_training_labels().shape))
  assert len(p.get_training_data()) == len(p.get_training_labels())

  logger.info('validation data shape: %s' % str(p.get_validation_data().shape))
  logger.info('validation label shape: %s' %
    str(p.get_validation_labels().shape))
  assert len(p.get_validation_data()) == len(p.get_validation_labels())

  logger.info('test data shape: %s' % str(p.get_test_images().shape))
  assert len(p.get_test_images()) == len(p.get_test_files())


if __name__ == '__main__':
  main()

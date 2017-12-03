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
      self._load()

  def _load(self):
    self.cursor = self.connection.cursor()
    self.cursor.execute("""SELECT * FROM meta;""")
    meta = self.cursor.fetchone()
    width = meta[0]
    height = meta[1]
    channel = meta[2]

    self.width = width
    self.height = height
    self.output_size = 12

    logger.info('loading images and labels...')
    self.cursor.execute("""SELECT image, classes FROM plants;""")
    raw_data = self.cursor.fetchall()

    self.images = []
    self.labels = []
    for d in raw_data:
      img = np.frombuffer(d[0], np.uint8)
      self.images.append(np.reshape(img, [height, width, channel]))
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
    self.training_data = self.images[training_index, :]
    self.training_label = self.labels[training_index, :]
    self.validation_data = self.images[validation_index, :]
    self.validation_label = self.labels[validation_index, :]

    self.images = self.images[index, :]
    self.labels = self.labels[index, :]

    self.cursor.execute("""SELECT fileName, image FROM test;""")
    raw_test_data = self.cursor.fetchall()
    self.test_files = []
    self.test_images = []
    for d in raw_test_data:
      img = np.frombuffer(d[1], np.uint8)
      self.test_files.append(d[0])
      self.test_images.append(np.reshape(img, [height, width, channel]))
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
  print(p.get_data().shape)
  print(p.get_label().shape)
  print(p.get_label_name())

  print(p.get_training_data().shape)
  print(p.get_training_labels().shape)

  print(p.get_validation_data().shape)
  print(p.get_validation_labels().shape)

  print(p.get_test_files())
  print(p.get_test_images().shape)


if __name__ == '__main__':
  main()

from urllib2 import urlopen
import logging
import os
import gzip
import struct
import numpy as np
import sqlite3
import unittest
from argparse import ArgumentParser


URL = 'http://yann.lecun.com/exdb/mnist/'
FILES = [
  'train-images-idx3-ubyte.gz',
  'train-labels-idx1-ubyte.gz',
  't10k-images-idx3-ubyte.gz',
  't10k-labels-idx1-ubyte.gz']


logging.basicConfig()
logger = logging.getLogger('mnist')
logger.setLevel(logging.INFO)


def download_files(folder):
  if not os.path.isdir(folder):
    os.mkdir(folder)

  for filename in FILES:
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):
      logger.info('downloading %s...', filename)
      url = URL + filename
      content = urlopen(url).read()
      with open(filepath, 'wb') as f:
        f.write(content)


def parse_image(dbname, tablename, image_path, label_path):
  logger.info('inserting data for %s...', tablename)
  connection = sqlite3.connect(dbname)
  cursor = connection.cursor()
  cursor.execute("""DROP TABLE IF EXISTS %s;""" % (tablename))
  cursor.execute("""CREATE TABLE %s(
    image BLOB NOT NULL,
    label INTEGER NOT NULL);""" % (tablename))

  with gzip.open(image_path, 'rb') as image_file:
    image_content = image_file.read()

    header = struct.unpack('>iiii', image_content[:16])
    assert header[0] == 2051

    count = header[1]
    height, width = header[2:]
    assert height == width
    assert width == 28
    size = height * width

    with gzip.open(label_path, 'rb') as label_file:
      label_content = label_file.read()

      header = struct.unpack('>ii', label_content[:8])
      assert header[0] == 2049
      assert header[1] == count

      for i in range(count):
        offset = 16 + size * i
        image_data = image_content[offset:offset+size]

        offset = 8 + i
        label_data = ord(label_content[offset:offset+1])
        cursor.execute("""INSERT INTO %s VALUES(?, ?)""" % (tablename),
          (buffer(image_data), label_data))
  connection.commit()
  connection.close()


class MNISTData(object):
  def __init__(self, dbname):
    if os.path.isfile(dbname):
      self.connection = sqlite3.connect(dbname)
      self.cursor = self.connection.cursor()
    else:
      logger.info('fail to load %s.', dbname)

    self._load_data()

  def _load_data(self):
    self.data = []
    self.label = []

    if hasattr(self, 'cursor'):
      self.data, self.label = self._load_from_table('train')
      self.test_data, self.test_label = self._load_from_table('test')

      index = np.random.permutation(np.arange(len(self.data)))
      self.data = self.data[index, :]
      self.label = self.label[index, :]

      self.training_data = self.data[:int(len(self.data) * 0.8)]
      self.training_label = self.label[:int(len(self.data) * 0.8)]

      self.validation_data = self.data[int(len(self.data) * 0.8):]
      self.validation_label = self.label[int(len(self.data) * 0.8):]

  def _load_from_table(self, table):
    data = []
    label = []
    self.cursor.execute("""SELECT * FROM %s;""" % (table))
    raw_data = self.cursor.fetchall()
    for entry in raw_data:
      img = np.frombuffer(entry[0], dtype=np.uint8).reshape([28, 28, 1])
      data.append(img)

      l = [0 for i in range(10)]
      l[entry[1]] = 1.0
      label.append(l)
    return np.array(data, dtype=np.uint8), np.array(label, dtype=np.uint8)

  def load_extra(self, extra_csv):
    if os.path.isfile(extra_csv):
      logger.info('loading extra data from %s...', extra_csv)
      extra_data = []
      extra_label = []
      with open(extra_csv, 'r') as f:
        f.readline()

        while True:
          line = f.readline().strip()
          if not line: break

          raw_entry = line.split(',')
          d = np.array([int(e) for e in raw_entry[1:]]).reshape([28, 28, 1])
          l = [1 if int(raw_entry[0]) == i else 0 for i in range(10)]
          extra_data.append(d)
          extra_label.append(l)

      self.training_data = np.concatenate([self.training_data, extra_data],
        axis=0)
      self.training_label = np.concatenate([self.training_label, extra_label],
        axis=0)


  def get_training_data(self):
    return self.training_data, self.training_label

  def get_validation_data(self):
    return self.validation_data, self.validation_label

  def get_test_data(self):
    return self.test_data, self.test_label


class TestMNISTData(unittest.TestCase):
  def setUp(self):
    self.mnist_data = MNISTData('mnist.sqlite3')

  def test_data_size(self):
    self.assertTrue(len(self.mnist_data.data) == 60000)
    self.assertTrue(len(self.mnist_data.label) == 60000)
    self.assertTrue(len(self.mnist_data.test_data) == 10000)
    self.assertTrue(len(self.mnist_data.test_label) == 10000)


def main():
  parser = ArgumentParser()
  parser.add_argument('--folder', dest='folder', default='data',
    type=str, help='folder to download mnist data')
  parser.add_argument('--dbname', dest='dbname', default='mnist.sqlite3',
    type=str, help='output sqlite3 db')
  args = parser.parse_args()

  download_files(args.folder)

  if not os.path.isfile(args.dbname):
    parse_image(args.dbname, 'train',
      os.path.join(args.folder, FILES[0]), os.path.join(args.folder, FILES[1]))
    parse_image(args.dbname, 'test',
      os.path.join(args.folder, FILES[2]), os.path.join(args.folder, FILES[3]))

  unittest.main()


if __name__ == '__main__':
  main()

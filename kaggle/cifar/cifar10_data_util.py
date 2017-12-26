import sqlite3
import numpy as np
import unittest


class CIFAR10Data(object):
  def __init__(self, dbname):
    self.connection = sqlite3.connect(dbname)
    self.cursor = self.connection.cursor()

    self._load()

  def _load_from_table(self, table):
    self.cursor.execute("""SELECT * FROM %s;""" % (table))
    raw_data = self.cursor.fetchall()

    data = []
    label = []
    for i in range(len(raw_data)):
      entry = raw_data[i]
      img = np.frombuffer(entry[0], np.uint8).reshape([32, 32, 3])
      l = np.array([1 if k == entry[1] else 0 for k in range(10)])
      data.append(img)
      label.append(l)
    return np.array(data), np.array(label)

  def _load(self):
    self.training_data, self.training_label = \
      self._load_from_table('cifar10_train')
    self.test_data, self.test_label = \
      self._load_from_table('cifar10_test')


class TestCIFAR10Data(unittest.TestCase):
  def setUp(self):
    self.data = CIFAR10Data('cifar.sqlite3')

  def test_data_size(self):
    self.assertTrue(len(self.data.training_data) == 50000)
    self.assertTrue(len(self.data.training_label) == 50000)

    self.assertTrue(len(self.data.test_data) == 10000)
    self.assertTrue(len(self.data.test_label) == 10000)


def main():
  unittest.main()


if __name__ == '__main__':
  main()

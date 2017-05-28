# -*- coding: utf-8 -*-

import sqlite3
import os
import logging
import numpy as np
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger('mnist_prepare')
logger.setLevel(logging.INFO)


def read_train_data(path):
  labels = []
  images = []
  if path and os.path.isfile(path):
    logger.info('reading %s' % (path))
    with open(path, 'r') as f:
      f.readline()
      while True:
        line = f.readline()
        if not line: break
        entry = line.split(',')
        label = int(entry[0])
        image = np.array([int(b) for b in entry[1:]], dtype=np.uint8)
        labels.append(label)
        images.append(image)
  else:
    logger.warning('%s not found' % (path))
  return images, labels


def read_test_data(path):
  images = []
  if path and os.path.isfile(path):
    logger.info('reading %s' % (path))
    with open(path, 'r') as f:
      f.readline()
      while True:
        line = f.readline()
        if not line: break
        entry = line.split(',')
        image = np.array([int(b) for b in entry], dtype=np.uint8)
        images.append(image)
  else:
    logger.warning('%s not found' % (path))
  return images


def save(train_data, train_label, test_data, name='mnist.sqlite3'):
  if not os.path.isfile(name):
    logger.info('saving into %s...' % (name))
    connection = sqlite3.connect(name)
    cursor = connection.cursor()
    # train data
    cursor.execute("""CREATE TABLE IF NOT EXISTS train(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL,
        label INTEGER NOT NULL);""")
    for i in range(len(train_data)):
      td = buffer(train_data[i])
      tl = train_label[i]
      cursor.execute("""INSERT INTO train(image, label) VALUES(?, ?);""",
          (td, tl,))
    # test data
    cursor.execute("""CREATE TABLE IF NOT EXISTS test(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL);""")
    for i in range(len(test_data)):
      td = buffer(test_data[i])
      cursor.execute("""INSERT INTO test(image) VALUES(?);""",
          (td,))
    connection.commit()
    connection.close()


def load(db):
  connection = sqlite3.connect(db)
  cursor = connection.cursor()
  cursor.execute("""SELECT image, label FROM train;""")
  train_data = cursor.fetchall()
  cursor.execute("""SELECT image FROM test;""")
  test_data = cursor.fetchall()
  connection.close()

  train_images = []
  train_labels = []
  test_images = []
  for entry in train_data:
    train_images.append(np.array(entry[0], dtype=np.uint8).reshape(28, 28, 1))
    train_labels.append([1 if l == entry[1] else 0 for l in range(10)])
  for entry in test_data:
    test_images.append(np.array(entry[0], dtype=np.uint8).reshape(28, 28, 1))
  return np.array(train_images), np.array(train_labels), np.array(test_images)


def main():
  parser = ArgumentParser()
  parser.add_argument('--train', dest='train', help='train data csv')
  parser.add_argument('--test', dest='test', help='test data csv')

  args = parser.parse_args()
  data, label = read_train_data(args.train)
  test_data = read_test_data(args.test)
  save(data, label, test_data)


if __name__ == '__main__':
  main()

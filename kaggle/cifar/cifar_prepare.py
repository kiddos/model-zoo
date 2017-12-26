from urllib2 import urlopen
import coloredlogs
import logging
import os
import md5
import tarfile
import cPickle
import sqlite3
import numpy as np
from argparse import ArgumentParser


URL = 'https://www.cs.toronto.edu/~kriz/'

coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('cifar')
logger.setLevel(logging.INFO)


def download_data(tar_file, folder, md5_value):
  try:
    if not os.path.isdir(folder):
      os.mkdir(folder)

    filepath = os.path.join(folder, tar_file)
    if not os.path.isfile(filepath):
      url = URL + tar_file
      logger.info('downloading %s from %s...', tar_file, URL)
      content = urlopen(url).read()

      with open(filepath, 'wb') as f:
        f.write(content)

    with open(filepath, 'rb') as f:
      value = md5.md5(f.read())
      assert value == md5_value
  except Exception as e:
    logger.error(e.message)


def extract(tar_file, folder, output_folder):
  if not os.path.isdir(os.path.join(folder, output_folder)):
    logger.info('extracting %s...', output_folder)
    with tarfile.open(os.path.join(folder, tar_file), 'r:gz') as f:
      f.extractall(folder)
  return os.path.join(folder, output_folder, 'train'), \
    os.path.join(folder, output_folder, 'test'), \
    os.path.join(folder, output_folder, 'meta')


def insert_cifar100(cursor, train_file, table):
  if os.path.isfile(train_file):
    with open(train_file, 'rb') as f:
      obj = cPickle.load(f)

      data = obj['data']
      fine_label = obj['fine_labels']
      coarse_label = obj['coarse_labels']

      for i in range(len(data)):
        data_entry = data[i]
        assert len(data_entry) == 32 * 32 * 3
        cursor.execute("""INSERT INTO %s VALUES(?, ?, ?)""" % (table),
          (buffer(data_entry), coarse_label[i], fine_label[i]))
  else:
    logger.error('%s not found', train_file)


def save_cifar100(dbname, folder):
  connection = sqlite3.connect(dbname)
  cursor = connection.cursor()

  logger.info('creating tables for cifar100...')
  cursor.execute("""DROP TABLE IF EXISTS cifar100_train;""")
  cursor.execute("""CREATE TABLE cifar100_train(
    image BLOB NOT NULL,
    coarseLabel INTEGER NOT NULL,
    fineLabel INTEGER NOT NULL);""")

  cursor.execute("""DROP TABLE IF EXISTS cifar100_test;""")
  cursor.execute("""CREATE TABLE cifar100_test(
    image BLOB NOT NULL,
    coarseLabel INTEGER NOT NULL,
    fineLabel INTEGER NOT NULL);""")

  logger.info('inserting training data...')
  insert_cifar100(cursor, os.path.join(folder, 'cifar-100-python', 'train'),
    'cifar100_train')

  logger.info('inserting test data...')
  insert_cifar100(cursor, os.path.join(folder, 'cifar-100-python', 'test'),
    'cifar100_test')
  connection.commit()
  connection.close()


def insert_cifar10(cursor, data_file, table):
  if os.path.isfile(data_file):
    with open(data_file, 'rb') as f:
      obj = cPickle.load(f)

      data = obj['data']
      label = obj['labels']

      for i in range(len(data)):
        data_entry = data[i]
        assert len(data_entry) == 32 * 32 * 3
        cursor.execute("""INSERT INTO %s VALUES(?, ?)""" % (table),
          (buffer(data_entry), label[i]))



def save_cifar10(dbname, folder):
  connection = sqlite3.connect(dbname)
  cursor = connection.cursor()

  logger.info('creating tables for cifar10...')
  cursor.execute("""DROP TABLE IF EXISTS cifar10_train;""")
  cursor.execute("""CREATE TABLE cifar10_train(
    image BLOB NOT NULL,
    label INTEGER NOT NULL);""")

  cursor.execute("""DROP TABLE IF EXISTS cifar10_test;""")
  cursor.execute("""CREATE TABLE cifar10_test(
    image BLOB NOT NULL,
    label INTEGER NOT NULL);""")

  logger.info('insert training data...')
  batches = ['data_batch_1', 'data_batch_2', 'data_batch_3',
    'data_batch_4', 'data_batch_5']
  for batch in batches:
    insert_cifar10(cursor, os.path.join(folder, batch), 'cifar10_train')
  connection.commit()

  logger.info('insert testing data...')
  insert_cifar10(cursor, os.path.join(folder, 'test_batch'), 'cifar10_test')
  connection.commit()
  connection.close()


def main():
  parser = ArgumentParser()
  parser.add_argument('--data-folder', dest='data_folder', default='data',
    type=str, help='folder to save data')
  parser.add_argument('--dbname', dest='dbname', default='cifar.sqlite3',
    type=str, help='output dbname')
  args = parser.parse_args()

  files = {
    'cifar-100-python.tar.gz': 'eb9058c3a382ffc7106e4002c42a8d85',
    'cifar-10-python.tar.gz': 'c58f30108f718f92721af3b95e74349a'}
  for f in files:
    download_data(f, args.data_folder, files[f])

  extract(files.keys()[0], args.data_folder, 'cifar-100-python')
  extract(files.keys()[1], args.data_folder, 'cifar-10-batches-py')

  save_cifar100(args.dbname, args.data_folder)
  save_cifar10(args.dbname, os.path.join(args.data_folder, 'cifar-10-batches-py'))


if __name__ == '__main__':
  main()

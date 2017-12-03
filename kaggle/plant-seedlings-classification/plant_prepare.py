import sqlite3
import os
import numpy as np
import logging
from argparse import ArgumentParser
from PIL import Image

logging.basicConfig()
logger = logging.getLogger('plants')
logger.setLevel(logging.INFO)


def load_data(train_folder, test_folder, dbname, width, height):
  connection = sqlite3.connect(dbname)
  cursor = connection.cursor()
  cursor.execute("""PRAGMA foreign_keys = ON;""")

  # creating tables
  logger.info('removing old data...')
  cursor.execute("""DROP TABLE IF EXISTS plants;""")
  cursor.execute("""DROP TABLE IF EXISTS test;""")
  cursor.execute("""DROP TABLE IF EXISTS classes;""")
  cursor.execute("""DROP TABLE IF EXISTS meta;""")

  logger.info('creating class table...')

  cursor.execute("""CREATE TABLE classes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    className TEXT NOT NULL);""")

  logger.info('creating plants table...')
  cursor.execute("""CREATE TABLE IF NOT EXISTS plants(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB NOT NULL,
    classes INTEGER REFERENCES classes(id));""")

  logger.info('creating test data...')
  cursor.execute("""CREATE TABLE test(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fileName TEXT NOT NULL,
    image BLOB NOT NULL);""")

  logger.info('creating meta table...')
  cursor.execute("""CREATE TABLE meta(
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    channel INTEGER NOT NULL);""")
  cursor.execute("""INSERT INTO meta VALUES(?, ?, ?);""",
    [width, height, 3])

  # loading train data
  class_id = 1
  data_folders = os.listdir(train_folder)
  for f in data_folders:
    class_name = f
    logger.info('loading %s images...' % class_name)
    cursor.execute("""INSERT INTO classes(className) VALUES(?);""",
      [class_name,])

    images = os.listdir(os.path.join(train_folder, f))
    for img in images:
      image_path = os.path.join(train_folder, f, img)
      i = Image.open(image_path).convert('RGB')
      i = i.resize([width, height])
      cursor.execute("""INSERT INTO plants(image, classes) VALUES(?, ?);""",
        [buffer(np.array(i)), class_id])
    class_id += 1

  # loading test data
  logger.info('loading test data...')
  for f in os.listdir(test_folder):
    image_path = os.path.join(test_folder, f)
    i = Image.open(image_path).convert('RGB')
    i = i.resize([width, height])
    cursor.execute("""INSERT INTO test(fileName, image) VALUES(?, ?);""",
      [f, buffer(np.array(i))])

  connection.commit()
  connection.close()


def main():
  parser = ArgumentParser()
  parser.add_argument('--train-folder', default='train', dest='train',
    type=str, help='folder to load images for training')
  parser.add_argument('--test-folder', default='test', dest='test',
    type=str, help='folder to load images for testing')
  parser.add_argument('--dbname', default='plants.sqlite3', dest='dbname',
    type=str, help='dbname to output')
  parser.add_argument('--width', default=64, dest='width',
    type=int, help='width of images')
  parser.add_argument('--height', default=64, dest='height',
    type=int, help='height of images')

  args = parser.parse_args()

  if os.path.isdir(args.train) and os.path.isdir(args.test):
    load_data(args.train, args.test, args.dbname, args.width, args.height)


if __name__ == '__main__':
  main()

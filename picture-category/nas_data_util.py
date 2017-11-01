# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import os
import sqlite3
import numpy as np
import logging
from PIL import Image
from argparse import ArgumentParser


IMAGE_WIDTH = 400
IMAGE_HEIGHT = 225
TABLE_NAME = 'jsn'
WIDTH_COLUMN = 'width'
HEIGHT_COLUMN = 'height'
IMAGE_DATA_COLUMN = 'imageData'
PATH_COLUMN = 'path'

logging.basicConfig()
logger = logging.getLogger('nas')
logger.setLevel(logging.INFO)


def load_paths(root_folder):
  pictures = []
  folders = os.listdir(root_folder)
  for item in folders:
    subpath = os.path.join(root_folder, item)
    if os.path.isdir(subpath):
      pictures += load_paths(subpath)
    elif os.path.isfile(subpath) and subpath.lower().endswith('.jpg'):
      pictures.append(subpath)
  return pictures


def store_images(paths, dbname):
  connection = sqlite3.connect(dbname)
  cursor = connection.cursor()
  cursor.execute("""CREATE TABLE IF NOT EXISTS %s(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    %s INTEGER NOT NULL,
    %s INTEGER NOT NULL,
    %s BLOB NOT NULL,
    %s BLOB NOT NULL);""" %
    (TABLE_NAME, WIDTH_COLUMN, HEIGHT_COLUMN, IMAGE_DATA_COLUMN, PATH_COLUMN))
  connection.commit()

  cursor.execute("""SELECT id from %s;""" % (TABLE_NAME))
  entry_count = len(cursor.fetchall())
  if entry_count == 0:
    for path in paths:
      img = np.array(Image.open(path).resize([IMAGE_WIDTH, IMAGE_HEIGHT]))
      buf = np.getbuffer(img)
      logger.info('%d. loading %s with %d bytes...' %
        (entry_count,path, len(buf)))
      cursor.execute("""INSERT INTO %s(%s, %s, %s, %s)
        VALUES(?, ?, ?, ?)""" % (TABLE_NAME, WIDTH_COLUMN, HEIGHT_COLUMN,
          IMAGE_DATA_COLUMN, PATH_COLUMN),
        (img.shape[1], img.shape[0], buf, buffer(path),))
      if entry_count % 10 == 0:
        connection.commit()
      entry_count += 1
  connection.commit()
  connection.close()


def main():
  parser = ArgumentParser()
  parser.add_argument('--root-path', dest='root_path',
    default='/media/nas/09活動照片/106學年度各項活動/106學年度上學期',
    type=str, help='image root path to load all paths for jpg images')
  parser.add_argument('--db-name', dest='dbname',
    default='jsn.sqlite3', type=str, help='db to store images')
  args = parser.parse_args()

  paths = load_paths(args.root_path)
  store_images(paths, args.dbname)


if __name__ == '__main__':
  main()

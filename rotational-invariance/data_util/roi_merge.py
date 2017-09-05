import sqlite3
import logging
import os
import numpy as np
import ast
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_entry(input_db, table_name):
  entries = None
  if os.path.isfile(input_db):
    logger.info('loading %s data...' % (input_db))
    connection = sqlite3.connect(input_db)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM %s;""" % (table_name))
    entries = cursor.fetchall()
    connection.close()
    return entries
  else:
    logger.info('%s not found' % (table_name))


def create_training_entry(entry, output_width, output_height):
  grid = np.zeros(shape=[output_height, output_width, 5], dtype=np.float32)
  count = np.zeros(shape=[output_height, output_width, 1])
  rects = ast.literal_eval(entry[5])
  for rect in rects:
    norm_cx = float(rect[0] + rect[2] / 2) / entry[2]
    norm_cy = float(rect[1] + rect[3] / 2) / entry[3]
    grid_x = int(norm_cx * output_width)
    grid_y = int(norm_cy * output_height)
    if count[grid_y, grid_x, 0] == 0.0:
      grid[grid_y, grid_x, 0] = 1.0
      grid[grid_y, grid_x, 1] = norm_cx
      grid[grid_y, grid_x, 2] = norm_cy
      grid[grid_y, grid_x, 3] = float(rect[2]) / entry[2]
      grid[grid_y, grid_x, 4] = float(rect[3]) / entry[3]
    else:
      a1 = grid[grid_y, grid_x, 3], grid[grid_y, grid_x, 4]
      norm_w = float(rect[2]) / entry[2]
      norm_h = float(rect[3]) / entry[3]
      a2 = norm_w * norm_h
      if a2 > a1:
        grid[grid_y, grid_x, 0] = 1.0
        grid[grid_y, grid_x, 1] = norm_cx
        grid[grid_y, grid_x, 2] = norm_cy
        grid[grid_y, grid_x, 3] = float(rect[2]) / entry[2]
        grid[grid_y, grid_x, 4] = float(rect[3]) / entry[3]
  return (entry[1], entry[2], entry[3], entry[4],
    buffer(grid), output_width, output_height, 5,)


def create_training_data(output_db, table_name,
    input_db, output_width, output_height):
  raw_entries = load_entry(input_db, table_name)
  if not raw_entries:
    return

  logger.info('creating output data...')
  connection = sqlite3.connect(output_db)
  cursor = connection.cursor()
  cursor.execute("""CREATE TABLE IF NOT EXISTS %s(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB NOT NULL,
    imageWidth INTEGER NOT NULL,
    imageHeight INTEGER NOT NULL,
    imageChannel INTEGER NOT NULL,
    grid BLOB NOT NULL,
    gridWidth INTEGER NOT NULL,
    gridHeight INTEGER NOT NULL,
    gridChannel INTEGER NOT NULL);""" % (table_name))
  logger.info('pushing data into sqlite3 db...')
  for raw_entry in raw_entries:
    cursor.execute("""INSERT INTO %s(image, imageWidth, imageHeight,
      imageChannel, grid, gridWidth, gridHeight, gridChannel)
      VALUES(?, ?, ?, ?, ?, ?, ?, ?);""" %
      (table_name), create_training_entry(raw_entry,
        output_width, output_height))
  logger.info('saving...')
  connection.commit()
  connection.close()


def main():
  parser = ArgumentParser()
  parser.add_argument('--output-db', dest='output_db',
    default='yolo.sqlite3', help='output roi sqlite3 dataset')
  parser.add_argument('--input-db', dest='input_db',
    default='roi.sqlite3', help='input roi dataset')
  parser.add_argument('--table-name', dest='table_name',
    default='roi', help='output table name')
  parser.add_argument('--grid-width', dest='grid_width',
    default=16, help='output grid width')
  parser.add_argument('--grid-height', dest='grid_height',
    default=12, help='output grid height')
  args = parser.parse_args()

  create_training_data(args.output_db, args.table_name,
    args.input_db, args.grid_width, args.grid_height)


if __name__ == '__main__':
  main()

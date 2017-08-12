import cv2
import os
import sqlite3
import logging
import numpy as np
from PIL import Image
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(db, table_name):
  if os.path.isfile(db):
    logger.info('loading %s...' % (db))
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM %s;""" % (table_name))
    data = cursor.fetchall()
    connection.close()
  else:
    logger.info('%s not found' % (db))
  return data


def main():
  parser = ArgumentParser()
  parser.add_argument('--db', dest='db',
    help='db to view')
  parser.add_argument('--table-name', dest='table_name',
    default='roi', help='output table name')
  parser.add_argument('--index', dest='index',
    default=0, type=int, help='index to start showing')
  args = parser.parse_args()

  entries = load_data(args.db, args.table_name)
  index = 0
  try:
    for i in range(args.index, len(entries)):
      logger.info('loading entry %d...' % (i))
      entry = entries[i]
      image = np.frombuffer(entry[1], dtype=np.uint8).reshape(
        [entry[3], entry[2], 3])
      cv2.imshow('image', image)
      key = cv2.waitKey(0)
      if key in [ord('q'), ord('Q'), 27]:
        break
      elif key in [ord('s'), ord('S')]:
        cv2.imwrite('image%d.jpg' % (index), image)
        index += 1
  except KeyboardInterrupt:
    logger.info('exit')



if __name__ == '__main__':
  main()

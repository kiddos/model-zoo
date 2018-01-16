import sqlite3
import os
import pandas as pd
import logging
import coloredlogs
import numpy as np
from PIL import Image
from argparse import ArgumentParser


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('prepare')
logger.setLevel(logging.INFO)


def load_labels(csv_file):
  labels = {}
  if os.path.isfile(csv_file):
    label = pd.read_csv(csv_file)
    for i, l in enumerate(label['Image']):
      labels[l] = label['Id'][i]
  return labels


def save_images(connection, folder, label, subsample=10):
  if os.path.isdir(folder):
    cursor = connection.cursor()

    logger.info('creating table...')
    cursor.execute("""CREATE TABLE IF NOT EXISTS images(
      image BLOB NOT NULL,
      width INTEGER NOT NULL,
      height INTEGER NOT NULL,
      label TEXT NOT NULL);""")

    image_files = os.listdir(folder)
    for i, image_file in enumerate(image_files):
      image_path = os.path.join(folder, image_file)
      image = Image.open(image_path).convert('L')
      img = np.copy(np.array(image)[::subsample, ::subsample])

      cursor.execute("""INSERT INTO images VALUES(?, ?, ?, ?)""",
        (np.getbuffer(img), img.shape[1], img.shape[0], label[image_file],))
      del image, img

      if i % (len(image_files) / 10) == 0:
        logger.info('%d/%d done.', i, len(image_files))
    logger.info('all done.')

    connection.commit()
    connection.close()
  else:
    logger.error('%s not found', folder)


def main():
  parser = ArgumentParser()
  parser.add_argument('--folder', dest='folder', default='train',
    help='folder where the images are')
  parser.add_argument('--dbname', dest='dbname',
    default='humback-whale.sqlite3', help='output sqlite3 dbname')
  parser.add_argument('--label-file', dest='label_file', default='train.csv',
    help='label file to load')
  parser.add_argument('--subsample', dest='subsample', default=10,
    type=int, help='image subsampling to reduce size')
  args = parser.parse_args()

  labels = load_labels(args.label_file)
  connection = sqlite3.connect(args.dbname)
  save_images(connection, args.folder, labels, subsample=args.subsample)


if __name__ == '__main__':
  main()

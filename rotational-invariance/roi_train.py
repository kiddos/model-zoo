import tensorflow as tf
import numpy as np
import os
import sqlite3
import logging
import cv2
from argparse import ArgumentParser

from roi import ROIModel


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(input_db, table_name,
    input_width, input_height):
  data = []
  label = []
  if os.path.isfile(input_db):
    logger.info('loading %s...' % (input_db))
    connection = sqlite3.connect(input_db)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM %s;""" % (table_name))
    raw_data = cursor.fetchall()
    for entry in raw_data:
      image = cv2.cvtColor(
        np.frombuffer(entry[1], dtype=np.uint8).reshape(
          [entry[3], entry[2], entry[4]]), cv2.COLOR_BGR2RGB)
      data.append(cv2.resize(image, (input_width, input_height)))
      grid = np.frombuffer(entry[5], dtype=np.float32).reshape(
        [entry[7], entry[6], entry[8]])
      label.append(grid)
  else:
    logger.info('%s not found' % (input_db))
  return np.array(data), np.array(label)


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-db', dest='input_db',
    default='roi.sqlite3', type=str, help='input roi dataset')
  parser.add_argument('--table-name', dest='table_name',
    default='roi', type=str, help='output table name')
  parser.add_argument('--input-width', dest='input_width',
    default=200, type=int, help='input image width')
  parser.add_argument('--input-height', dest='input_height',
    default=150, type=int, help='input image height')

  parser.add_argument('-e', dest='max_epoch', default=60000,
    type=int, help='max epoch for training')
  parser.add_argument('--batch-size', dest='batch_size', default=256,
    type=int, help='batch size for training')
  parser.add_argument('--output-period', dest='output_period', default=100,
    type=int, help='output period')
  parser.add_argument('--keep-prob', dest='keep_prob', default=0.8,
    type=int, help='keep probability for dropout')
  args = parser.parse_args()

  image, label = load_data(args.input_db, args.table_name,
    args.input_width, args.input_height)
  logger.info('image shape: %s' % (str(image.shape)))
  logger.info('label shape: %s' % (str(label.shape)))

  logger.info('initializing model...')
  model = ROIModel(args.input_width, args.input_height,
    image.shape[3], label.shape[2], label.shape[1])

  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    model.train(sess, image,
      label[:, :, :, 0:1],
      label[:, :, :, 1:3],
      label[:, :, :, 3:5],
      batch_size=args.batch_size,
      output_period=args.output_period,
      keep_prob=args.keep_prob,
      max_epoch=args.max_epoch)


if __name__ == '__main__':
  main()

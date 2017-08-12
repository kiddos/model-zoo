import tensorflow as tf
import numpy as np
import logging
import os
import time
from argparse import ArgumentParser

from tiny_yolo_v2 import TinyYOLOV2
from data_util.tiny_yolo_util import OUTPUT_WIDTH, OUTPUT_HEIGHT
from data_util.tiny_yolo_util import TinyYOLODataBatch


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-width', dest='input_width', default=240,
    type=int, help='input image width')
  parser.add_argument('--input-height', dest='input_height', default=180,
    type=int, help='input image height')
  parser.add_argument('--input-channel', dest='input_channel', default=1,
    type=int, help='input image channel')
  parser.add_argument('-i', dest='input_db',
    default='data_util/tiny_yolo_v2.sqlite3',
    help='input dataset')
  parser.add_argument('--table-name', dest='table_name',
    default='blocks', help='input dataset table name')

  parser.add_argument('-e', dest='max_epoch', default=100000,
    type=int, help='max epoch for training')
  parser.add_argument('--batch-size', dest='batch_size', default=256,
    type=int, help='batch size for training')
  parser.add_argument('--output-period', dest='output_period', default=100,
    type=int, help='output period')
  parser.add_argument('--keep-prob', dest='keep_prob', default=0.8,
    type=int, help='keep probability for dropout')
  args = parser.parse_args()

  batcher = TinyYOLODataBatch(args.input_db, args.table_name,
    args.input_width, args.input_height)
  model = TinyYOLOV2(args.input_width, args.input_height,
    args.input_channel, saving=True)

  training_config = tf.ConfigProto()
  training_config.gpu_options.allocator_type = 'BFC'
  training_config.gpu_options.allow_growth = True
  with tf.Session(config=training_config) as sess:
    model.train(sess, batcher,
      batch_size=args.batch_size,
      output_period=args.output_period,
      keep_prob=args.keep_prob,
      max_epoch=args.max_epoch)


if __name__ == '__main__':
  main()

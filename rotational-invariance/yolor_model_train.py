from __future__ import print_function

import tensorflow as tf
from argparse import ArgumentParser
import os

from data_util.yolo_data_util import load_yolo_data
from yolor_model import YOLORotationModel


def main():
  parser = ArgumentParser()
  parser.add_argument('-i', dest='input_db',
    default='yolo.sqlite3', type=str, help='input dataset to load')
  parser.add_argument('--table-name', dest='table_name',
    default='yolo', type=str, help='sqlite data table name to load')

  parser.add_argument('--input-width', dest='input_width',
    default=160, type=int, help='input image width')
  parser.add_argument('--input-height', dest='input_height',
    default=120, type=int, help='input image height')
  parser.add_argument('--output-width', dest='output_width',
    default=16, type=int, help='output image width')
  parser.add_argument('--output-height', dest='output_height',
    default=12, type=int, help='output image height')

  parser.add_argument('-b', '--batch-size', dest='batch_size',
    default=100, type=int, help='batch size for training')
  parser.add_argument('-m', '--max-epoch', dest='max_epoch',
    default=100000, type=int, help='max epoch to train model')
  parser.add_argument('--output-period', dest='output_period',
    default=10, type=int, help='epoch to output')
  parser.add_argument('--decay-epoch', dest='decay_epoch',
    default=1000, type=int, help='epoch to decay learning rate')
  parser.add_argument('--keep-prob', dest='keep_prob',
    default=0.8, type=float, help='keep probability')

  parser.add_argument('--checkpoint', dest='checkpoint',
    default='', type=str, help='checkpoint to load and continue training')

  args = parser.parse_args()

  print('loading data...')
  images, labels = load_yolo_data(args.input_db, args.table_name)
  print('training data shapes: %s' % (str(images.shape)))
  print('training label shapes: %s' % (str(labels.shape)))

  model = YOLORotationModel(
    args.input_width, args.input_height, 3,
    args.output_width, args.output_height, 1, labels.shape[3] - 6,
    model_name='YOLOWithRotation', saving=True)


  config = tf.ConfigProto()
  with tf.Session(config=config) as sess:
    if args.checkpoint != '' and \
        os.path.isfile(args.checkpoint + '.meta') and \
        os.path.isfile(args.checkpoint + '.index'):
      print('loading checkpoint %s...' % (args.checkpoint))
      model.load(sess, args.checkpoint)
    print('start training...')
    model.train(sess,
      images,
      labels[:, :, :, 0:1],
      labels[:, :, :, 1:3],
      labels[:, :, :, 3:5],
      labels[:, :, :, 5:6],
      labels[:, :, :, 6:],
      batch_size=args.batch_size,
      output_period=args.output_period,
      keep_prob=args.keep_prob,
      decay_epoch=args.decay_epoch,
      max_epoch=args.max_epoch)


if __name__ == '__main__':
  main()

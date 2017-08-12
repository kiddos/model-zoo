from __future__ import print_function
from optparse import OptionParser
import tensorflow as tf
import logging

from data_util.yolo_data_util import load_data, CHARACTERS
from yolo_model import YOLOModel


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
  parser = OptionParser()
  parser.add_option('-c', '--checkpoint', dest='checkpoint', default='',
      help='last checkpoint to load for further training')
  parser.add_option('-n', '--model_name', dest='model_name', default='YOLO',
      help='model name for output')
  parser.add_option('-i', '--input_db', dest='input_db',
      default='yolo_train.sqlite',
      help='model name for output')
  parser.add_option('-e', '--max_epoch', dest='max_epoch', default=10000,
      help='max epoch for training')
  parser.add_option('-b', '--batch_size', dest='batch_size', default=1,
      help='batch size for training')
  parser.add_option('-k', '--output_width', dest='output_width', default=16,
      help='output width')
  parser.add_option('-j', '--output_height', dest='output_height', default=12,
      help='output height')
  parser.add_option('-d', '--output_period', dest='output_period', default=10,
      help='output period for training')
  options, args = parser.parse_args()

  output_width = int(options.output_width)
  output_height = int(options.output_height)
  images, coordinates, dimensions, classes = \
    load_data(options.input_db, CHARACTERS,
      output_width, output_height, 1)
  model = YOLOModel(160, 120, 3,
      output_width, output_height, 1,
      model_name=options.model_name)

  with tf.Session() as sess:
    if options.checkpoint:
      model.load(sess, options.checkpoint)
    try:
      model.train(sess, images, coordinates, dimensions, classes,
        batch_size=int(options.batch_size),
        output_period=int(options.output_period),
        max_epoch=int(options.max_epoch))
    except KeyboardInterrupt:
      logger.info('stop training.')


if __name__ == '__main__':
  main()

from optparse import OptionParser
import numpy as np
import tensorflow as tf
import os
import logging

from data_util.rotational_invariance_data_util import load_data, CHARACTERS
from rotation_invariance_model import RotationalInvarianceModel


logging.basicConfig()
logger = logging.getLogger('test_rotation_invariance_model')
logger.setLevel(logging.INFO)


def main():
  parser = OptionParser()
  parser.add_option('-c', '--checkpoint', dest='checkpoint', default='',
      help='last checkpoint to load for further training')
  parser.add_option('-n', '--model_name', dest='model_name',
      default='RotationalInvarianceModel',
      help='model name for output')
  parser.add_option('-e', '--max_epoch', dest='max_epoch', default=30000,
      help='max epoch for training')
  parser.add_option('-b', '--batch_size', dest='batch_size', default=256,
      help='batch size for training')
  parser.add_option('-p', '--output_period', dest='output_period',
      default=1000,
      help='output period for training')
  parser.add_option('-d', '--decay_epoch', dest='decay_epoch', default=1000,
      help='the epoch when learning rate decay')
  parser.add_option('-k', '--keep_prob', dest='keep_prob', default=0.8,
      help='keep probability for dropout layers')
  parser.add_option('-v', '--variance', dest='variance', default=1.0,
      help='variance for input data')
  parser.add_option('-i', '--input_db', dest='input_db',
      default='characters_train.sqlite',
      help='input database path')
  options, args = parser.parse_args()

  logger.info('preparing data...')
  data, label = load_data(options.input_db, CHARACTERS)
  rotations = label[:, -1:]
  label = label[:, :-1]
  model = RotationalInvarianceModel(64, 3, 10, model_name=options.model_name)

  if not os.path.isfile(options.input_db):
    logger.warning('%s does not exists' % (options.input_db))
    return

  logger.info('starting tf Session...')
  with tf.Session() as sess:
    if options.checkpoint:
      logger.info('load checkpoing %s' % (options.checkpoint))
      model.load(sess, options.checkpoint)
    try:
      model.train(sess, data, label, rotations,
        batch_size=int(options.batch_size),
        output_period=int(options.output_period),
        max_epoch=int(options.max_epoch),
        decay_epoch=int(options.decay_epoch),
        keep_prob=float(options.keep_prob),
        variance=float(options.variance))
    except KeyboardInterrupt:
      logger.info('stop training.')


if __name__ == '__main__':
  main()

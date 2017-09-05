from optparse import OptionParser
import numpy as np
import tensorflow as tf
import os
import logging

from data_util import load_data, CHARACTERS
from rotation_invariance_model import RotationalInvarianceModel


logging.basicConfig()
logger = logging.getLogger('test_rotation_invariance_model')
logger.setLevel(logging.INFO)


def compute_accuracy(prediction, answer):
  return np.sum(np.argmax(prediction, axis=1) ==
      np.argmax(answer, axis=1)) * 100.0 / len(prediction)


def compute_error(prediction, answer):
  return np.sum(np.abs(prediction- answer)) / len(prediction)


def main():
  parser = OptionParser()
  parser.add_option('-c', '--checkpoint', dest='checkpoint', default='',
      help='last checkpoint to load for testing')
  parser.add_option('-n', '--model_name', dest='model_name',
      default='RotationalInvarianceModel',
      help='model name for output')
  parser.add_option('-b', '--batch_size', dest='batch_size', default=1000,
      help='batch size for training')
  parser.add_option('-i', '--input_db', dest='input_db',
      default='characters_test.sqlite',
      help='input database path')
  options, args = parser.parse_args()

  if not os.path.isfile(options.input_db):
    logger.warning('%s does not exists' % (options.input_db))
    return

  if not os.path.isfile(options.checkpoint + '.index'):
    logger.warning('%s checkpoint does not exists' % (options.input_db))
    return

  logger.info('preparing data...')
  data, label = load_data(options.input_db, CHARACTERS)
  rotations = label[:, -1:]
  label = label[:, :-1]
  model = RotationalInvarianceModel(64, 3, 10, model_name=options.model_name)

  logger.info('starting tf Session...')
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load(sess, options.checkpoint)
    try:
      batch_size = int(options.batch_size)
      num_batches = len(data) / batch_size

      accuracy = 0
      error = 0
      for n in range(num_batches):
        batch_data = data[n:n+batch_size, :]
        batch_label = label[n:n+batch_size, :]
        batch_rotation = rotations[n:n+batch_size, :]

        classes, rotate = model.predict(sess, batch_data)
        accuracy += compute_accuracy(classes, batch_label)
        error += compute_error(rotate, batch_rotation)
      accuracy /= num_batches
      error /= num_batches
      logger.info('Test Accuracy: %f | Test Rotation Error: %f' %
          (accuracy, error))
    except KeyboardInterrupt:
      logger.info('stop testing.')


if __name__ == '__main__':
  main()

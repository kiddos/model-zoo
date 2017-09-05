import tensorflow as tf
import numpy as np
import sqlite3
import logging
import os
from argparse import ArgumentParser

from mnist_prepare import load
from mnist import MNISTConvolutionModel


IMAGE_SIZE = 28

logging.basicConfig()
logger = logging.getLogger('mnist predict')
logger.setLevel(logging.INFO)


def load_test_data(db_path):
  data = []
  if os.path.isfile(db_path):
    logger.info('loading %s...' % (db_path))
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM test;""")
    raw_data = cursor.fetchall()
    connection.close()
    for entry in raw_data:
      data.append(np.reshape(np.frombuffer(entry[1], np.uint8),
        [IMAGE_SIZE, IMAGE_SIZE, 1]))
  else:
    logger.info('%s does not exists' % (db_path))
  return np.array(data)


def main():
  parser = ArgumentParser()
  parser.add_argument('--db', dest='db', default='mnist.sqlite3',
    help='input sqlite3 db')
  parser.add_argument('--checkpoint', dest='checkpoint',
    default='MNIST1/MNIST-100000', help='checkpoint name')
  parser.add_argument('--output-csv', dest='output',
    default='output.csv', help='output csv name')

  args = parser.parse_args()
  test_data = load_test_data(args.db)
  logger.info('test data shape: %s' % (str(test_data.shape)))

  if os.path.isfile(args.checkpoint + '.data-00000-of-00001') and \
      os.path.isfile(args.checkpoint + '.meta') and \
      os.path.isfile(args.checkpoint + '.index'):
    logger.info('setting up model...')
    model = MNISTConvolutionModel(IMAGE_SIZE, IMAGE_SIZE, 1, 10,
      saving=False, model_name='MNIST')
    saver = tf.train.Saver()

    parts = 100
    with tf.Session() as sess:
      logger.info('restoring checkpoint...')
      saver.restore(sess, args.checkpoint)

      logger.info('writing result...')
      with open(args.output, 'w') as f:
        f.write('ImageId,Label\n')
        for i in range(0, len(test_data), parts):
          result = np.argmax(
            model.predict(sess, test_data[i:(i+parts), :]), axis=1)
          logger.info('result: %s' % (str(result)))
          for j in range(len(result)):
            f.write('%d,%d\n' % (i + j + 1, result[j]))
  else:
    logger.info('no valid checkpoint found.')


if __name__ == '__main__':
  main()

import tensorflow as tf
import numpy as np
import os
import logging
import pandas
from argparse import ArgumentParser

from titanic_data_util import analyse_data, parse_data


logging.basicConfig()
logger = logging.getLogger('titanic')
logger.setLevel(logging.INFO)


def load_graph(sess, checkpoint):
  meta = checkpoint + '.meta'
  if os.path.isfile(meta):
    saver = tf.train.import_meta_graph(meta, clear_devices=True)

    if os.path.isfile(checkpoint + '.data-00000-of-00001'):
      saver.restore(sess, checkpoint)
    else:
      logger.error('Fail to load %s', checkpoint)
  else:
    logger.error('meta file not found')

  graph = tf.get_default_graph()
  return graph


def run(sess, graph, test_file, output_file):
  inputs = graph.get_tensor_by_name('inputs:0')
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  output = graph.get_tensor_by_name('titanic/outputs/prediction:0')

  if os.path.isfile(test_file):
    test_data = pandas.read_csv(test_file)

    analyse_data(test_data)
    ids = test_data['PassengerId']
    data = parse_data(test_data)

    output = sess.run(output, feed_dict={
      inputs: data,
      keep_prob: 1.0,
    })

    with open(output_file, 'w') as f:
      f.write('PassengerId,Survived\n')

      for i, result in enumerate(output):
        f.write('%d,%d\n' % (ids[i], np.argmax(result)))


def main():
  parser = ArgumentParser()
  parser.add_argument('--checkpoint', dest='checkpoint',
    help='checkpoint to load')
  parser.add_argument('--test-file', dest='test_file',
    default='test.csv', help='test csv file')
  parser.add_argument('--output-file', dest='output_file',
    default='titanic.csv', help='output file')
  args = parser.parse_args()

  if hasattr(args, 'checkpoint'):
    with tf.Session() as sess:
      graph = load_graph(sess, args.checkpoint)
      run(sess, graph, args.test_file, args.output_file)


if __name__ == '__main__':
  main()

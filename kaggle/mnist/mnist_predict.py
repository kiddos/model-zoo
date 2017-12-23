import tensorflow as tf
import numpy as np
import sqlite3
import logging
import os
from argparse import ArgumentParser

from mnist_prepare import MNISTData


logging.basicConfig()
logger = logging.getLogger('mnist predict')
logger.setLevel(logging.INFO)


def load_graph(sess, checkpoint):
  if os.path.isfile(checkpoint + '.meta'):
    saver = tf.train.import_meta_graph(checkpoint + '.meta')

    if os.path.isfile(checkpoint + '.data-00000-of-00001'):
      saver.restore(sess, checkpoint)
  else:
    logger.warning('checkpoint %s not valid', checkpoint)
  return tf.get_default_graph()


def save_graph(sess, graph, output_nodes, output_file):
  input_graph_def = graph.as_graph_def()
  output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, input_graph_def, output_nodes)
  logger.info('final output graph: %d nodes.', len(output_graph_def.node))

  with tf.gfile.GFile(output_file, 'wb') as f:
    logger.info('writing into %s...', output_file)
    f.write(output_graph_def.SerializeToString())


def load_test_data(test_csv):
  test_data = []
  if os.path.isfile(test_csv):
    logger.info('loading %s...', test_csv)
    with open(test_csv, 'r') as f:
      f.readline()

      while True:
        line = f.readline().strip()
        if not line: break
        entry = np.array([int(e) for e in line.split(',')])
        entry = entry.reshape([28, 28, 1])
        test_data.append(entry)
  return np.array(test_data)


def accuracy(prediction, label):
  p = np.argmax(prediction, axis=1)
  l = np.argmax(label, axis=1)
  return np.sum((p == l).astype(np.float32)) / len(prediction)


def predict(args):
  with tf.Session() as sess:
    graph = load_graph(sess, args.checkpoint)
    input_images = graph.get_tensor_by_name('inputs/input_images:0')
    keep_prob = graph.get_tensor_by_name('inputs/keep_prob:0')
    prediction = graph.get_tensor_by_name('mnist/output/prediction:0')

    mnist_data = MNISTData(args.dbname)
    test_data, test_label = mnist_data.get_test_data()

    a = 0
    for i in range(0, len(test_data), args.batch_size):
      test_batch = test_data[i:i+args.batch_size, :]
      label_batch = test_label[i:i+args.batch_size, :]

      p = sess.run(prediction, feed_dict={
        input_images: test_batch,
        keep_prob: 1.0,
      })
      a += accuracy(p, label_batch)
    logger.info('accuarcy: %f', a / (len(test_data) / args.batch_size))

    test_data = load_test_data(args.test_csv)
    with open(args.output_csv, 'w') as f:
      logger.info('writing to %s...', args.output_csv)
      f.write('ImageId,Label\n')

      for i in range(len(test_data)):
        p = sess.run(prediction, feed_dict={
          input_images: test_data[i:i+1, :],
          keep_prob: 1.0,
        })
        f.write('%d,%d\n' % (i + 1, np.argmax(p)))

    if args.saving == 'True':
      save_graph(sess, graph, args.output_nodes.split(','),
        args.output_file)


def main():
  parser = ArgumentParser()
  parser.add_argument('--dbname', dest='dbname', default='mnist.sqlite3',
    type=str, help='test sqlite3 db data')
  parser.add_argument('--checkpoint', dest='checkpoint',
    type=str, help='checkpoint name')
  parser.add_argument('--test-csv', dest='test_csv', default='test.csv',
    type=str, help='test data to load')

  parser.add_argument('--saving', dest='saving',
    default='False', type=str, help='save model')
  parser.add_argument('--output-file', dest='output_file',
    default='mnist.pb', help='saving graph file')
  parser.add_argument('--output-nodes', dest='output_nodes',
    default='mnist/prediction', help='output node to save to')
  parser.add_argument('--batch-size', dest='batch_size',
    default=100, type=int, help='batch size to predict')

  parser.add_argument('--output-csv', dest='output_csv',
    default='mnist.csv', help='output csv file')

  args = parser.parse_args()
  if hasattr(args, 'checkpoint'):
    predict(args)
  else:
    logger.error('no checkpoint specified')


if __name__ == '__main__':
  main()

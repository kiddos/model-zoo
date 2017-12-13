import tensorflow as tf
import numpy as np
import os
import logging
from PIL import Image
from argparse import ArgumentParser

logging.basicConfig()
logger = logging.getLogger('yolo')
logger.setLevel(logging.INFO)


def load_graph(sess, checkpoint):
  meta = checkpoint + '.meta'
  if os.path.isfile(meta):
    logger.info('loading graph from %s...', meta)
    saver = tf.train.import_meta_graph(meta, clear_devices=True)

    if os.path.isfile(checkpoint + '.data-00000-of-00001'):
      logger.info('restoring session from %s...', checkpoint)
      saver.restore(sess, checkpoint)
      return saver
    else:
      logger.error('fail to restore session from %s' % (checkpoint))
  else:
    logger.error('fail to load graph from %s.' % (meta))


def freeze_graph(sess, output_nodes, output_graph):
  input_graph = tf.get_default_graph()
  input_graph_def = input_graph.as_graph_def()
  output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, input_graph_def, output_nodes.split(','))
  logger.info('%d ops in final graph', len(output_graph_def.node))

  with tf.gfile.GFile(output_graph, 'wb') as f:
    logger.info('writing to %s...', output_graph)
    f.write(output_graph_def.SerializeToString())
  return output_graph_def


def main():
  parser = ArgumentParser()
  parser.add_argument('--checkpoint', dest='checkpoint', type=str,
    help='checkpoint to load and freeze')
  parser.add_argument('--output-nodes', dest='output_nodes', type=str,
    default='yolo/concat', help='output node name')
  parser.add_argument('--output-graph', dest='output_graph', type=str,
    default='yolo.pb', help='output graph name')

  args = parser.parse_args()

  if hasattr(args, 'checkpoint') and args.checkpoint:
    if not os.path.isfile(args.output_graph):
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(config=config) as sess:
        load_graph(sess, args.checkpoint)
        freeze_graph(sess, args.output_nodes, args.output_graph)
  else:
    logger.warning('need to specify checkpoint')


if __name__ == '__main__':
  main()

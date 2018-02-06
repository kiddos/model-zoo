import tensorflow as tf
import os
import logging


logging.basicConfig()
logger = logging.getLogger('freeze')
logger.setLevel(logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint', 'model-100000',
  'saved checkpoint to load')
tf.app.flags.DEFINE_string('output_nodes',
  'inference/outputs/prediction', 'output nodes to save')
tf.app.flags.DEFINE_string('output_model', 'humback-whale.pb',
  'output model')


def freeze_model():
  meta = FLAGS.checkpoint + '.meta'
  if os.path.isfile(meta) and \
      os.path.isfile(FLAGS.checkpoint + '.data-00000-of-00001'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      logger.info('importing meta graph...')
      saver = tf.train.import_meta_graph(meta)
      saver.restore(sess, FLAGS.checkpoint)

      logger.info('converting variables to constants...')
      graph = tf.get_default_graph()
      input_graph_def = graph.as_graph_def()
      output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, input_graph_def, FLAGS.output_nodes.split(','))

      with tf.gfile.GFile(FLAGS.output_model, 'wb') as gf:
        logger.info('saving model %s...', FLAGS.output_model)
        gf.write(output_graph_def.SerializeToString())
      return graph
  else:
    logger.error('Fail to load %s', FLAGS.checkpoint)


def main(_):
  freeze_model()


if __name__ == '__main__':
  tf.app.run()

import tensorflow as tf
import os
import logging


logging.basicConfig()
logger = logging.getLogger('freeze')
logger.setLevel(logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint', 'breakout',
  'checkpoint to freeze')
tf.app.flags.DEFINE_string('output_nodes', 'train/output/q_values',
  'output nodes to save')
tf.app.flags.DEFINE_string('output_model', 'breakout.pb',
  'output freezed model')


def load_graph():
  meta = FLAGS.checkpoint + '.meta'
  data = FLAGS.checkpoint + '.data-00000-of-00001'
  if os.path.isfile(meta) and os.path.isfile(data):
    saver = tf.train.import_meta_graph(meta, clear_devices=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      logger.info('restoring session...')
      saver.restore(sess, FLAGS.checkpoint)

      logger.info('converting variables to constant...')
      input_graph_def = tf.get_default_graph().as_graph_def()
      nodes = FLAGS.output_nodes.split(',')
      output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, input_graph_def, nodes)

      with tf.gfile.GFile(FLAGS.output_model, 'wb') as gf:
        logger.info('saving model to %s...', FLAGS.output_model)
        gf.write(output_graph_def.SerializeToString())
  else:
    logger.error('Fail to find %s.', FLAGS.checkpoint)


def main(_):
  load_graph()


if __name__ == '__main__':
  tf.app.run()

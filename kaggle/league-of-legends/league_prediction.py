import tensorflow as tf
import numpy as np
import os
import logging
import sqlite3
import sys


logging.basicConfig()
logger = logging.getLogger('league')
logger.setLevel(logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('save', False, 'save model')
tf.app.flags.DEFINE_string('dbname', 'league.sqlite3', 'extracted data to load')

# hyperparameter
tf.app.flags.DEFINE_integer('embedding_size', 3, 'embedding size to use')
tf.app.flags.DEFINE_float('lambda_reg', 3e-4, 'parameter for regularization')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'learning rate to train')
tf.app.flags.DEFINE_integer('max_epoch', 10000, 'max epoch to train')
tf.app.flags.DEFINE_integer('batch_size', 512, 'batch size to train')

tf.app.flags.DEFINE_integer('display_epoch', 100, 'epoch to display result')
tf.app.flags.DEFINE_integer('summary_epoch', 10, 'epoch to save summary')
tf.app.flags.DEFINE_string('output_nodes', 'output/prediction', 'output nodes')


def load_data(dbname, tablename):
  if os.path.isfile(dbname):
    connection = sqlite3.connect(dbname)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM %s;""" % (tablename))
    raw_data = cursor.fetchall()
    raw_data = np.array(raw_data)

    cursor.execute("""SELECT count(id) FROM champs;""")
    champ_count = cursor.fetchone()

    connection.close()
    return raw_data[:, :-2], raw_data[:, -2:], champ_count[0]
  else:
    logger.info('%s does not exists', dbname)


class EmbeddedModel(object):
  def __init__(self, champ_count):
    self.champ_count = champ_count

    self._setup_inputs()
    logits, self.prediction = self._inference(self.champs)

    reg = 0
    var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in var:
      shape = v.get_shape().as_list()
      if len(shape) > 1 and shape[0] != 140 and \
          shape[1] != FLAGS.embedding_size:
        reg += tf.reduce_sum(tf.square(v))

    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=self.match_result))
      self.loss = tf.add(self.loss, reg * FLAGS.lambda_reg)

    with tf.name_scope('optimizer'):
      self.learning_rate = tf.Variable(FLAGS.learning_rate,
        name='learning_rate')
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

    with tf.name_scope('evaluation'):
      self.accuracy = self.evaluate(self.prediction, self.match_result)

  def _setup_inputs(self):
    self.champs = tf.placeholder(dtype=tf.int32, shape=[None, 10],
      name='champs')
    self.match_result = tf.placeholder(dtype=tf.float32, shape=[None, 2],
      name='match_result')

  def _inference(self, inputs):
    init = tf.random_normal_initializer(stddev=1.0)
    with tf.name_scope('embeddings'):
      embeddings = tf.get_variable('embeddings', shape=[self.champ_count,
        FLAGS.embedding_size], initializer=init)

    init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope('fully_connected'):
      fc = tf.nn.embedding_lookup(embeddings, self.champs)
      fc = tf.contrib.layers.flatten(fc)

      for _ in range(1):
        fc = tf.contrib.layers.fully_connected(fc, 128,
          weights_initializer=init)

    with tf.name_scope('output'):
      ow = tf.get_variable('ow', shape=[128, 2], initializer=init)
      ob = tf.get_variable('ob', shape=[2], initializer=tf.zeros_initializer())
      logits = tf.add(tf.matmul(fc, ow), ob, name='logits')
      outputs = tf.nn.softmax(logits, name='prediction')
    return logits, outputs

  def evaluate(self, prediction, labels):
    count = tf.reduce_mean(tf.cast(tf.equal(
      tf.argmax(prediction, axis=1), tf.argmax(labels, axis=1)), tf.float32))
    return count

  def train(self, train_data, train_label, valid_data, valid_label):
    dataset = tf.data.Dataset.from_tensor_slices({
      'champs': train_data,
      'match_result': train_label
    })
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_initializable_iterator()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())

      try:
        epoch = 0
        for epoch in range(FLAGS.max_epoch + 1):
          sess.run(iterator.initializer)

          batch = sess.run(iterator.get_next())
          if epoch % FLAGS.display_epoch == 0:
            loss = sess.run(self.loss, feed_dict={
              self.champs: batch['champs'],
              self.match_result: batch['match_result'],
            })

            train_acc = sess.run(self.accuracy, feed_dict={
              self.champs: batch['champs'],
              self.match_result: batch['match_result'],
            })
            valid_acc = sess.run(self.accuracy, feed_dict={
              self.champs: valid_data,
              self.match_result: valid_label
            })
            logger.info('%d. loss: %f, train: %f, valid: %f',
              epoch, loss, train_acc, valid_acc)

          sess.run(self.train_ops, feed_dict={
            self.champs: batch['champs'],
            self.match_result: batch['match_result'],
          })
      except KeyboardInterrupt:
        logger.info('interrupted')
      finally:
        logger.info('saving model...')
        input_graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
          sess, input_graph_def, FLAGS.output_nodes.split(','))
        with tf.gfile.GFile('league.pb', 'wb') as f:
          f.write(output_graph_def.SerializeToString())


def main():
  logger.info('loading %s', FLAGS.dbname)
  train_data, train_label, champ_count = load_data(FLAGS.dbname, 'matches_train')
  valid_data, valid_label, _ = load_data(FLAGS.dbname, 'matches_valid')
  model = EmbeddedModel(champ_count)

  model.train(train_data, train_label, valid_data, valid_label)


if __name__ == '__main__':
  main()

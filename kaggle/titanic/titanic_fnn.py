import tensorflow as tf
import numpy as np
import logging
import os
import pandas as pd
from argparse import ArgumentParser

from titanic_data_util import analyse_data, parse_data, parse_label


logging.basicConfig()
logger = logging.getLogger('titanic')
logger.setLevel(logging.INFO)


class Titanic(object):
  def __init__(self, input_size, output_size, learning_rate=1e-3,
      decay=0.9, lambda_reg=1e-4):
    self.input_size = input_size
    self.output_size = output_size

    self._setup_inputs()
    with tf.variable_scope('titanic'):
      logits, self.output = self.inference(self.inputs)

    with tf.variable_scope('titanic', reuse=True):
      self.valid_logits, self.valid_output = self.inference(self.valid_inputs)

    with tf.name_scope('loss'):
      train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'titanic')
      regularization = lambda_reg * tf.reduce_sum(tf.square(train_vars[0]))
      for i in range(1, len(train_vars)):
        regularization += lambda_reg * tf.reduce_sum(tf.square(train_vars[i]))

      sigma = tf.Variable(1.0, name='variance')
      variance = tf.square(sigma)
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=self.labels), name='loss')
      self.loss += regularization

      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('variance', variance)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(learning_rate, trainable=False)
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)
      self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * decay)
      tf.summary.scalar('learning_rate', self.learning_rate)

    with tf.name_scope('evaluation'):
      self.accuray = self.evaluate(self.output, self.labels)
      self.valid_accuracy = self.evaluate(self.valid_output, self.valid_labels)
      tf.summary.scalar('accuarcy', self.accuray)
      tf.summary.scalar('validation_accuarcy', self.valid_accuracy)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _setup_inputs(self):
    self.inputs = tf.placeholder(dtype=tf.float32, name='inputs',
      shape=[None, self.input_size])
    self.labels = tf.placeholder(dtype=tf.float32, name='labels',
      shape=[None, self.output_size])
    self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob',
      shape=[])

    self.valid_inputs = tf.placeholder(dtype=tf.float32, name='valid_inputs',
      shape=[None, self.input_size])
    self.valid_labels = tf.placeholder(dtype=tf.float32, name='valid_labels',
      shape=[None, self.output_size])

  def preprocess(self, inputs):
    noise_generator = tf.random_normal_initializer(mean=0.0, stddev=0.3)
    noise = noise_generator([self.input_size])
    preprocessed = inputs + noise
    return preprocessed

  def inference(self, inputs):
    with tf.name_scope('fc1'):
      fc = tf.contrib.layers.fully_connected(inputs, 256,
        activation_fn=tf.nn.tanh,
        weights_initializer=tf.random_normal_initializer(stddev=1.0))

    with tf.name_scope('norm1'):
      norm = tf.layers.batch_normalization(fc)

    with tf.name_scope('fc2'):
      fc = self.dense(norm, 256, multiple=1)

    with tf.name_scope('drop2'):
      drop = tf.nn.dropout(fc, keep_prob=self.keep_prob)

    with tf.name_scope('outputs'):
      logits = tf.contrib.layers.fully_connected(drop, self.output_size,
        activation_fn=None)
      output = tf.nn.softmax(logits, name='prediction')
    return logits, output

  def evaluate(self, prediction, labels):
    p = tf.argmax(prediction, axis=1)
    l = tf.argmax(labels, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(p, l), tf.float32))
    return accuracy

  def prepare_folder(self):
    index = 0
    folder = 'titanic_fnn_%d' % index
    while os.path.isdir(folder):
      index += 1
      folder = 'titanic_fnn_%d' % index
    os.mkdir(folder)
    return folder

  def dense(self, inputs, size, multiple=1):
    fc = tf.contrib.layers.fully_connected(inputs, size,
      activation_fn=tf.nn.tanh,
      weights_initializer=tf.variance_scaling_initializer())
    for i in range(multiple):
      fc = tf.contrib.layers.fully_connected(fc, size,
        activation_fn=tf.nn.tanh,
        weights_initializer=tf.variance_scaling_initializer())
    return fc


def load_data(csv_files, training_percent=0.8):
  data = []
  label = []
  for csv_file in csv_files:
    if os.path.isfile(csv_file):
      titanic_data = pd.read_csv(csv_file)
      analyse_data(titanic_data, plot=False)

      d = parse_data(titanic_data)
      l = parse_label(titanic_data)
      data.append(d)
      label.append(l)

  data = np.concatenate(data, axis=0)
  label = np.concatenate(label, axis=0)

  data_size = len(data)
  index = np.random.permutation(np.arange(data_size))

  training_size = int(data_size * training_percent)
  train_index = index[:training_size]
  train_data, train_label = data[train_index, :], label[train_index, :]
  valid_index = index[training_size:]
  valid_data, valid_label = data[valid_index, :], label[valid_index, :]
  return train_data, train_label, valid_data, valid_label


def train(args):
  train_data, train_label, valid_data, valid_label = \
    load_data(args.csv_files.split(','))

  if args.load_all == 'True':
    train_data = np.concatenate([train_data, valid_data], axis=0)
    train_label = np.concatenate([train_label, valid_label], axis=0)

  logger.info('training data size: %d', len(train_data))
  logger.info('validation data size: %d', len(valid_data))

  model = Titanic(train_data.shape[1], train_label.shape[1],
    learning_rate=args.learning_rate, decay=args.decay)

  if args.saving:
    folder = model.prepare_folder()
    checkpoint = os.path.join(folder, 'titanic')

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
      tf.get_default_graph())

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initializing variables...')
    sess.run(tf.global_variables_initializer())

    data_size = len(train_data)
    for epoch in range(args.max_epoches + 1):
      indices = np.random.permutation(np.arange(data_size))[:args.batch_size]
      batch_data = train_data[indices, :]
      batch_label = train_label[indices, :]

      if epoch % args.display_epoch == 0:
        loss, accuracy, valid_accuracy, valid_logits = sess.run(
          [model.loss, model.accuray, model.valid_accuracy, model.valid_logits],
          feed_dict={
            model.inputs: batch_data,
            model.labels: batch_label,
            model.valid_inputs: valid_data,
            model.valid_labels: valid_label,
            model.keep_prob: 1.0
          })
        logger.info('%d. loss: %f, accuracy: %f, validation: %f',
          epoch, loss, accuracy, valid_accuracy)

      sess.run(model.train_ops, feed_dict={
        model.inputs: batch_data,
        model.labels: batch_label,
        model.keep_prob: args.keep_prob,
      })

      if epoch % args.save_epoch == 0 and args.saving:
        saver.save(sess, checkpoint, global_step=epoch)

        summary = sess.run(model.summary, feed_dict={
          model.inputs: batch_data,
          model.labels: batch_label,
          model.valid_inputs: valid_data,
          model.valid_labels: valid_label,
          model.keep_prob: 1.0
        })
        summary_writer.add_summary(summary, global_step=epoch)

      if epoch % args.decay_epoch == 0 and epoch != 0:
        logger.info('decay learning rate...')
        sess.run(model.decay_lr)


def main():
  parser = ArgumentParser()

  parser.add_argument('--load-all', dest='load_all', default='False',
    type=str, help='load all data to train')

  parser.add_argument('--mode', dest='mode', default='train',
    type=str, help='train/test')

  parser.add_argument('--csv-files', dest='csv_files',
    default='titanic_clean.csv,train.csv', help='training csv file')
  parser.add_argument('--extra-csv-file', dest='extra_csv_file',
    default='train.csv', type=str, help='extra training data')

  parser.add_argument('--saving', dest='saving', default=False,
    type=bool, help='rather to save model or not')

  parser.add_argument('--learning-rate', dest='learning_rate', default=3e-3,
    type=float, help='learning rate to train model')
  parser.add_argument('--max-epoches', dest='max_epoches', default=100000,
    type=int, help='max epoches to train model')
  parser.add_argument('--display-epoches', dest='display_epoch', default=100,
    type=int, help='epoches to evaluation')
  parser.add_argument('--save-epoches', dest='save_epoch', default=20000,
    type=int, help='epoches to save model')
  parser.add_argument('--decay-epoch', dest='decay_epoch', default=20000,
    type=int, help='epoches to decay learning rate')
  parser.add_argument('--batch-size', dest='batch_size', default=128,
    type=int, help='batch size to train model')
  parser.add_argument('--keep-prob', dest='keep_prob', default=0.75,
    type=float, help='keep probability for dropout')
  parser.add_argument('--decay', dest='decay', default=0.9,
    type=float, help='decay learning rate')

  args = parser.parse_args()

  if args.mode == 'train':
    train(args)


if __name__ == '__main__':
  main()

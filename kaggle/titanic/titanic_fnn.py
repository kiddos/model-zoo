import tensorflow as tf
import numpy as np
import logging
import os
from argparse import ArgumentParser

from titanic_prepare import load_data

logging.basicConfig()
logger = logging.getLogger('titanic')
logger.setLevel(logging.INFO)


class Titanic(object):
  def __init__(self, input_size, output_size, learning_rate=1e-3):
    self.input_size = input_size
    self.output_size = output_size

    self._setup_inputs()
    with tf.variable_scope('fnn'):
      logits, self.output = self.inference(self.inputs)

    with tf.variable_scope('fnn', reuse=True):
      _, self.valid_output = self.inference(self.valid_inputs)

    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=self.labels), name='loss')
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(learning_rate, trainable=False)
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)
      self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * 0.9)
      tf.summary.scalar('learning_rate', self.learning_rate)

    with tf.name_scope('evaluation'):
      self.accuray = self.evaluate(self.output, self.labels)
      self.valid_accuracy = self.evaluate(self.valid_output, self.valid_labels)
      tf.summary.scalar('accuarcy', self.accuray)
      tf.summary.scalar('validation_accuarcy', self.valid_accuray)

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

  def inference(self, inputs):
    with tf.name_scope('fc1'):
      fc = tf.contrib.layers.fully_connected(inputs, 64,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('norm1'):
      norm = tf.layers.batch_normalization(fc)

    with tf.name_scope('fc2'):
      fc = tf.contrib.layers.fully_connected(norm, 128,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('fc3'):
      fc = tf.contrib.layers.fully_connected(fc, 256,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('outputs'):
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
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


def train(args):
  model = Titanic(9, 2, learning_rate=args.learning_rate)

  training_data, training_label, valid_data, valid_label = \
    load_data(args.csv_file)
  data_size = len(training_data)
  logger.info('training data size: %d', len(training_data))
  logger.info('validation data size: %d', len(valid_data))

  if args.load_all:
    training_data = np.concatenate([training_data, valid_data], axis=1)
    training_label = np.concatenate([training_label, valid_label], axis=1)

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

    valid_index = 0
    for epoch in range(args.max_epoches + 1):
      offset = epoch % (data_size - args.batch_size)
      to = offset + args.batch_size
      data_batch = training_data[offset:to, :]
      label_batch = training_label[offset:to, :]

      if epoch % args.display_epoch == 0:
        valid_data_batch = \
          valid_data[valid_index:valid_index+args.batch_size, :]
        valid_label_batch = \
          valid_label[valid_index:valid_index+args.batch_size, :]
        loss, accuracy, valid_accuracy = sess.run(
          [model.loss, model.accuray, model.valid_accuracy],
          feed_dict={
            model.inputs: data_batch,
            model.labels: label_batch,
            model.valid_inputs: valid_data_batch,
            model.valid_labels: valid_label_batch,
            model.keep_prob: 1.0
          })
        logger.info('%d. loss: %f, accuracy: %f, validation: %f',
          epoch, loss, accuracy, valid_accuracy)

        valid_index = (valid_index + args.batch_size) % (len(valid_data) -
          args.batch_size)

      sess.run(model.train_ops, feed_dict={
        model.inputs: data_batch,
        model.labels: label_batch,
        model.keep_prob: args.keep_prob,
      })

      if epoch % args.save_epoch == 0:
        saver.save(sess, checkpoint, global_step=epoch)

        summary = sess.run(model.summary, feed_dict={
          model.inputs: data_batch,
          model.labels: label_batch,
          model.valid_inputs: valid_data_batch,
          model.valid_labels: valid_label_batch,
          model.keep_prob: 1.0
        })
        summary_writer.add_summary(summary, global_step=epoch)


def main():
  parser = ArgumentParser()

  parser.add_argument('--load-all', dest='load_all', default=False,
    type=bool, help='load all data to train')

  parser.add_argument('--mode', dest='mode', default='train',
    type=str, help='train/test')

  parser.add_argument('--csv-file', dest='csv_file', default='train.csv',
    type=str, help='training csv file')

  parser.add_argument('--saving', dest='saving', default=False,
    type=bool, help='rather to save model or not')

  parser.add_argument('--learning-rate', dest='learning_rate', default=1e-4,
    type=float, help='learning rate to train model')
  parser.add_argument('--max-epoches', dest='max_epoches', default=60000,
    type=int, help='max epoches to train model')
  parser.add_argument('--display-epoches', dest='display_epoch', default=100,
    type=int, help='epoches to evaluation')
  parser.add_argument('--save-epoches', dest='save_epoch', default=1000,
    type=int, help='epoches to save model')
  parser.add_argument('--decay-epoch', dest='decay_epoch', default=10000,
    type=int, help='epoches to decay learning rate')
  parser.add_argument('--batch-size', dest='batch_size', default=128,
    type=int, help='batch size to train model')
  parser.add_argument('--keep-prob', dest='keep_prob', default=0.8,
    type=float, help='keep probability for dropout')

  args = parser.parse_args()

  if args.mode == 'train':
    train(args)


if __name__ == '__main__':
  main()

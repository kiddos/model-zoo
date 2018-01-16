import tensorflow as tf
import numpy as np
import logging
import time
import os
from argparse import ArgumentParser

from mnist_prepare import MNISTData


logging.basicConfig()
logger = logging.getLogger('mnist')
logger.setLevel(logging.INFO)


class MNIST(object):
  def __init__(self, inference, learning_rate=1e-3, saving=True):
    with tf.name_scope('inputs'):
      self._setup_inputs()

    tf.summary.image(name='input_images', tensor=self.input_images)
    tf.summary.histogram(name='labels', values=self.labels)

    self.inference_name = inference

    if hasattr(self, inference):
      inference_fn = getattr(self, inference)

    with tf.variable_scope('mnist'):
      logits, self.output = inference_fn(self.input_images)

    with tf.variable_scope('mnist', reuse=True):
      _, self.validation_output = inference_fn(self.validation_inputs)

    with tf.name_scope('evaluation'):
      self.accuracy = self.evaluate(self.output, self.labels)
      self.validation_accuarcy = self.evaluate(self.validation_output,
        self.validation_label)

      tf.summary.scalar('train_accuarcy', self.accuracy)
      tf.summary.scalar('validation_accuracy', self.validation_accuarcy)

    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=self.labels))
    tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimizer'):
      self.learning_rate = tf.Variable(learning_rate, trainable=False,
        name='learning_rate')
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

      self.decay_lr = tf.assign(self.learning_rate,
        self.learning_rate * 0.66, name='decay_learning_rate')

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def prepare_folder(self):
    index = 0
    folder = 'mnist_%s_%d' % (self.inference_name, index)
    while os.path.isdir(folder):
      index += 1
      folder = 'mnist_%s_%d' % (self.inference_name, index)
    os.mkdir(folder)
    return folder

  def _setup_inputs(self):
    self.input_images = tf.placeholder(dtype=tf.float32, name='input_images',
      shape=[None, 28, 28, 1])
    self.labels = tf.placeholder(dtype=tf.float32, name='labels',
      shape=[None, 10])
    self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

    self.validation_inputs = tf.placeholder(dtype=tf.float32, name='input_images',
      shape=[None, 28, 28, 1])
    self.validation_label = tf.placeholder(dtype=tf.float32, name='labels',
      shape=[None, 10])

  def inference_v0(self, inputs):
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=1, kernel_size=3,
        weights_initializer=tf.random_normal_initializer(stddev=0.05))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = self.multiple_conv(pool, 64, 3)

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = self.multiple_conv(pool, 128, 3)

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv4'):
      conv = self.multiple_conv(pool, 256, 3)

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop4'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), 1024,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      logits = tf.contrib.layers.fully_connected(fc, 10,
        activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
      outputs = tf.nn.softmax(logits, name='prediction')
    return logits, outputs

  def inference_v1(self, inputs):
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 64, stride=1, kernel_size=3,
        weights_initializer=tf.random_normal_initializer(stddev=0.05))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = self.multiple_conv(pool, 128, 3)

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = self.multiple_conv(pool, 256, 3)

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv4'):
      conv = self.multiple_conv(pool, 512, 3)

    with tf.name_scope('drop4'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), 1024,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      logits = tf.contrib.layers.fully_connected(fc, 10,
        activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
      outputs = tf.nn.softmax(logits, name='prediction')
    return logits, outputs

  def inference_v2(self, inputs):
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 256, stride=1, kernel_size=7,
        weights_initializer=tf.random_normal_initializer(stddev=0.05))

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(conv, 256, stride=1, kernel_size=5,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(conv, 1024, stride=1, kernel_size=5,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(pool, 1024, stride=1, kernel_size=5,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop4'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), 1024,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      logits = tf.contrib.layers.fully_connected(fc, 10,
        activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
      outputs = tf.nn.softmax(logits, name='prediction')
    return logits, outputs

  def inference_v3(self, inputs):
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 256, stride=1, kernel_size=5,
        weights_initializer=tf.random_normal_initializer(stddev=0.05))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 384, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(conv, 384, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(conv, 384, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv5'):
      conv = tf.contrib.layers.conv2d(pool, 512, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv6'):
      conv = tf.contrib.layers.conv2d(pool, 512, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv7'):
      conv = tf.contrib.layers.conv2d(pool, 512, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool7'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv8'):
      conv = tf.contrib.layers.conv2d(pool, 1024, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv9'):
      conv = tf.contrib.layers.conv2d(conv, 1024, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv10'):
      conv = tf.contrib.layers.conv2d(conv, 1024, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool10'):
      pool = tf.contrib.layers.max_pool2d(conv, 3)

    with tf.name_scope('conv11'):
      conv = tf.contrib.layers.conv2d(pool, 1024, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('drop11'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('output'):
      logits = tf.contrib.layers.conv2d(drop, 10, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())
      logits = tf.reshape(logits, [-1, 10])
      outputs = tf.nn.softmax(logits, name='prediction')
    return logits, outputs

  def multiple_conv(self, inputs, size, ksize, multiple=1):
    conv = tf.contrib.layers.conv2d(inputs, size, stride=1, kernel_size=ksize,
      weights_initializer=tf.variance_scaling_initializer())
    for i in range(multiple):
      conv = tf.contrib.layers.conv2d(conv, size / 2, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())
      conv = tf.contrib.layers.conv2d(conv, size, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())
    return conv

  def evaluate(self, prediction, label):
    p = tf.argmax(prediction, axis=1)
    l = tf.argmax(label, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(p, l), tf.float32))
    return accuracy


def train(args):
  mnist_data = MNISTData(args.dbname)

  training_data, training_label = mnist_data.get_training_data()
  validation_data, validation_label = mnist_data.get_validation_data()

  if args.load_all == 'True':
    mnist_data.load_extra(args.extra_csv)
    training_data, training_label = mnist_data.get_training_data()
    training_data = np.concatenate([training_data, validation_data], axis=0)
    training_label = np.concatenate([training_label, validation_label], axis=0)

  model = MNIST(args.inference, learning_rate=args.learning_rate)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  saving = args.saving == 'True'
  if saving:
    folder = model.prepare_folder()
    checkpoint = os.path.join(folder, 'mnist')
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
      tf.get_default_graph())

  with tf.Session(config=config) as sess:
    logger.info('initializing variables...')
    sess.run(tf.global_variables_initializer())

    validation_index = 0
    logger.info('start training...')
    logger.info('training data: %s', str(training_data.shape))
    logger.info('training label: %s', str(training_label.shape))

    training_size = len(training_data)
    total_time = 0
    offset = 0
    for epoch in range(args.max_epoches + 1):
      # preprare data
      data_batch = training_data[offset:offset+args.batch_size, :]
      label_batch = training_label[offset:offset+args.batch_size, :]
      offset += args.batch_size
      if offset >= training_size - args.batch_size and offset < training_size:
        offset = training_size - args.batch_size
      elif offset >= training_size:
        offset = 0

      if epoch % args.display_epoches == 0:
        offset = validation_index
        to = validation_index + args.batch_size

        validation_data_batch = validation_data[offset:to, :]
        validation_label_batch = validation_label[offset:to, :]

        loss, accuarcy, validation_accuracy = sess.run(
          [model.loss, model.accuracy, model.validation_accuarcy],
          feed_dict={
            model.input_images: data_batch,
            model.labels: label_batch,
            model.validation_inputs: validation_data_batch,
            model.validation_label: validation_label_batch,
            model.keep_prob: args.keep_prob,
          })
        logger.info('%d. loss: %f, accuracy: %f, validation: %f',
          epoch, loss, accuarcy, validation_accuracy)

        validation_index = (validation_index +
          args.batch_size) % (len(validation_data) - args.batch_size)

        ave = total_time / (epoch + 1)
        time_remaining = (args.max_epoches - epoch) * ave
        days = int(time_remaining / 86400)
        time_remaining %= 86400
        hours = int(time_remaining / 3600)
        time_remaining %= 3600
        minutes = int(time_remaining / 60)
        time_remaining %= 60
        seconds = int(time_remaining)
        logger.info('time remaining: %d days %d hours %d minutes %d seconds' %
          (days, hours, minutes, seconds))

      # train
      start_time = time.time()
      sess.run(model.train_ops, feed_dict={
        model.input_images: data_batch,
        model.labels: label_batch,
        model.keep_prob: args.keep_prob,
      })
      passed = time.time() - start_time
      total_time += passed

      if epoch % args.save_epoches == 0 and epoch != 0 and saving:
        logger.info('saving model...')
        saver.save(sess, checkpoint, global_step=epoch)

      if epoch % args.summary_epoches == 0 and epoch != 0 and saving:
        summary = sess.run(model.summary, feed_dict={
          model.input_images: data_batch,
          model.labels: label_batch,
          model.validation_inputs: validation_data_batch,
          model.validation_label: validation_label_batch,
          model.keep_prob: 1.0,
        })
        summary_writer.add_summary(summary, global_step=epoch)

      if epoch % args.decay_epoches == 0 and epoch != 0:
        logger.info('decay learning rate...')
        sess.run(model.decay_lr)


def main():
  parser = ArgumentParser()
  parser.add_argument('--dbname', dest='dbname', default='mnist.sqlite3',
    type=str, help='sqlite3 db to load for training')

  parser.add_argument('--inference', dest='inference', default='inference_v0',
    type=str, help='inference model version')

  parser.add_argument('--learning-rate', dest='learning_rate', default=1e-3,
    type=float, help='learning rate to train model')
  parser.add_argument('--keep-prob', dest='keep_prob', default=0.8,
    type=float, help='keep probability for dropouts')
  parser.add_argument('--max-epoches', dest='max_epoches', default=100000,
    type=int, help='max epoches to train model')
  parser.add_argument('--save-epoches', dest='save_epoches', default=10000,
    type=int, help='epoches to save model')
  parser.add_argument('--display-epoches', dest='display_epoches', default=100,
    type=int, help='epoches to display')
  parser.add_argument('--decay-epoches', dest='decay_epoches', default=5000,
    type=int, help='epoches to decay learning rate')
  parser.add_argument('--summary-epoches', dest='summary_epoches', default=100,
    type=int, help='epoches to save summary for model')
  parser.add_argument('--batch-size', dest='batch_size', default=32,
    type=int, help='batch size for training')
  parser.add_argument('--saving', dest='saving', default='False',
    type=str, help='save model')

  parser.add_argument('--load-all', dest='load_all', default='False',
    type=str, help='load all data')
  parser.add_argument('--extra_csv', dest='extra_csv', default='train.csv',
    type=str, help='extra training data to load')
  args = parser.parse_args()

  train(args)


if __name__ == '__main__':
  main()

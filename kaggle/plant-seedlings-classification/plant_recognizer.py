import tensorflow as tf
import numpy as np
import logging
import os
import time
import sys
from argparse import ArgumentParser

from plant_loader import PlantLoader


logging.basicConfig()
logger = logging.getLogger('plants')
logger.setLevel(logging.INFO)


class PlantRecognizer(object):
  def __init__(self, inference,
      learning_rate, input_width, input_height, output_size):
    self.learning_rate = tf.Variable(learning_rate, trainable=False,
      name='learning_rate')
    if learning_rate != 0:
      self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * 0.9)

    self.input_width = input_width
    self.input_height = input_height
    self.output_size = output_size

    if hasattr(self, inference):
      inference_func = getattr(self, inference)
      self.inference = inference
    else:
      logger.error('no such inference function!!')
      sys.exit(1)

    self._setup_inputs()

    with tf.variable_scope('inference'):
      logits, self.outputs = inference_func(self.inputs)

    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,
        labels=self.labels))
      tf.summary.scalar(name='loss', tensor=self.loss)

    with tf.name_scope('optimization'):
      self.global_step = tf.contrib.framework.get_or_create_global_step()
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss,
        global_step=self.global_step)

    with tf.variable_scope('inference', reuse=True):
      _, self.validation_outputs = inference_func(self.validation_inputs)

    with tf.name_scope('evalutation'):
      self.validation_accuracy = self.evaluate(self.validation_outputs,
        self.validation_labels)
      self.accuracy = self.evaluate(self.outputs, self.labels)

      tf.summary.scalar(name='validation_accuracy',
        tensor=self.validation_accuracy)
      tf.summary.scalar(name='training_accuracy', tensor=self.accuracy)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def evaluate(self, outputs, labels):
    prediction = tf.argmax(outputs, axis=1)
    answer = tf.argmax(labels, axis=1)
    accuracy = tf.reduce_mean(
      tf.cast(tf.equal(prediction, answer), tf.float32)) * 100.0
    return accuracy

  def _setup_inputs(self):
    with tf.device('/:cpu0'):
      self.inputs = tf.placeholder(dtype=tf.float32, name='image_inputs',
        shape=[None, self.input_height, self.input_width, 3])
      self.labels = tf.placeholder(dtype=tf.float32, name='labels',
        shape=[None, self.output_size])
      self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob',
        shape=[])

      self.validation_inputs = tf.placeholder(dtype=tf.float32,
        name='valid_inputs',
        shape=[None, self.input_height, self.input_width, 3])
      self.validation_labels = tf.placeholder(dtype=tf.float32,
        name='valid_labels', shape=[None, self.output_size])
      tf.summary.image(name='input', tensor=self.inputs)
      tf.summary.image(name='validation_input', tensor=self.validation_inputs)

  def inference_v0(self, inputs):
    stddev = 0.03
    ksize = 7
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 64, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, kernel_size=2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 128, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, kernel_size=2)

    with tf.name_scope('fully_connected'):
      connect_shape = pool.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      stddev = np.sqrt(2.0 / connect_size)
      fc_size = 4096
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(pool, [-1, connect_size]), fc_size,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('output'):
      stddev = np.sqrt(2.0 / fc_size)
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
      outputs = tf.nn.softmax(logits)
    return logits, outputs

  def inference_v1(self, inputs):
    stddev = 0.05
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=1, kernel_size=3,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 64, stride=1, kernel_size=3,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, kernel_size=2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 128, stride=1, kernel_size=3,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, kernel_size=2)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(conv, 256, stride=1, kernel_size=3,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('drop4'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      stddev = np.sqrt(2.0 / connect_size)
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), 4096,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
      connect_size = 4094

    with tf.name_scope('output'):
      stddev = np.sqrt(2.0 / connect_size)
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
      outputs = tf.nn.softmax(logits)
    return logits, outputs

  def inference_v2(self, inputs):
    stddev = 0.05
    ksize = 3
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 4, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 8, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, kernel_size=2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 16, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, kernel_size=2)

    with tf.name_scope('drop3'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      stddev = np.sqrt(2.0 / connect_size)
      fc_size = 2048
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), fc_size,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('output'):
      stddev = np.sqrt(2.0 / fc_size)
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
      outputs = tf.nn.softmax(logits)
    return logits, outputs

  def inference_v3(self, inputs):
    stddev = 0.066
    ksize = 3
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 4, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(conv, 4, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 8, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(conv, 8, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, kernel_size=2)

    with tf.name_scope('conv5'):
      conv = tf.contrib.layers.conv2d(pool, 16, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('conv6'):
      conv = tf.contrib.layers.conv2d(conv, 16, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool6'):
      pool = tf.contrib.layers.max_pool2d(conv, kernel_size=2)

    with tf.name_scope('drop6'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc_size = 2048
      stddev = np.sqrt(2.0 / connect_size)
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), fc_size,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('output'):
      stddev = np.sqrt(2.0 / fc_size)
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
      outputs = tf.nn.softmax(logits)
    return logits, outputs

  def inference_v4(self, inputs):
    stddev = 0.09
    ksize = 3
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(conv, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 64, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(conv, 64, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop4'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      stddev = np.sqrt(2.0 / connect_size)
      fc_size = 512
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), fc_size,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('output'):
      stddev = np.sqrt(2.0 / fc_size)
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
      outputs = tf.nn.softmax(logits)
    return logits, outputs

  def inference_v5(self, inputs):
    stddev = 0.05
    ksize = 3
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('drop1'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(drop, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 64, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('drop3'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(drop, 64, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv5'):
      conv = tf.contrib.layers.conv2d(pool, 128, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('drop5'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('conv6'):
      conv = tf.contrib.layers.conv2d(drop, 128, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool6'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('fully_connected'):
      connect_shape = pool.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      stddev = np.sqrt(2.0 / connect_size)
      fc_size = 512
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(pool, [-1, connect_size]), fc_size,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('output'):
      stddev = np.sqrt(2.0 / fc_size)
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
      outputs = tf.nn.softmax(logits)
    return logits, outputs

  def inference_v6(self, inputs):
    ksize = 3
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 16, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('drop1'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(drop, 16, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())

      conv = tf.contrib.layers.conv2d(conv, 8, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())

      conv = tf.contrib.layers.conv2d(conv, 16, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())

      conv = tf.contrib.layers.conv2d(conv, 16, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())

      conv = tf.contrib.layers.conv2d(conv, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('drop3'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(drop, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())

      conv = tf.contrib.layers.conv2d(conv, 16, stride=1, kernel_size=1,
        weights_initializer=tf.variance_scaling_initializer())

      conv = tf.contrib.layers.conv2d(conv, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv5'):
      conv = tf.contrib.layers.conv2d(pool, 128, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('drop5'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('conv6'):
      conv = tf.contrib.layers.conv2d(drop, 128, stride=1, kernel_size=ksize,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool6'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('fully_connected'):
      connect_shape = pool.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(pool, [-1, connect_size]), 256,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
      outputs = tf.nn.softmax(logits)
    return logits, outputs

  def prepare_folder(self):
    index = 0
    folder = 'plant-recognizer-%s_%d' % (self.inference, index)
    while os.path.isdir(folder):
      index += 1
      folder = 'plant-recognizer-%s_%d' % (self.inference, index)
    os.mkdir(folder)
    return folder


def train(dbname, args):
  loader = PlantLoader(dbname)
  loader.load_data()

  recognizer = PlantRecognizer(args.inference, args.learning_rate,
    loader.get_width(), loader.get_height(),
    loader.get_output_size())

  if args.load_all:
    training_data = loader.get_data()
    training_labels = loader.get_label()
  else:
    training_data = loader.get_training_data()
    training_labels = loader.get_training_labels()
  logger.info(training_data.shape)
  logger.info(training_labels.shape)

  validation_data = loader.get_validation_data()
  validation_labels = loader.get_validation_labels()

  if args.saving:
    saver = tf.train.Saver()
    folder = recognizer.prepare_folder()
    checkpoint = os.path.join(folder, 'plant-recognizer')

    summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
      graph=tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    data_size = len(training_data)
    batch_size = args.batch_size
    total_time = 0
    for epoch in range(args.max_epoches + 1):
      offset = epoch % (data_size - batch_size)

      data_batch = training_data[offset:offset+batch_size, :]
      label_batch = training_labels[offset:offset+batch_size, :]
      #  data_batch = validation_data[offset:offset+batch_size, :]
      #  label_batch = validation_labels[offset:offset+batch_size, :]

      if epoch % args.display_epoch == 0:
        o = epoch % (len(validation_data) - batch_size)
        tensor = [recognizer.loss, recognizer.accuracy,
          recognizer.validation_accuracy]
        loss, train_accuracy, valid_accuarcy = sess.run(tensor,
          feed_dict={
            recognizer.inputs: data_batch,
            recognizer.labels: label_batch,
            recognizer.validation_inputs: validation_data[o:o+batch_size, :],
            recognizer.validation_labels: validation_labels[o:o+batch_size, :],
            recognizer.keep_prob: 1.0,
          })
        ave = 0
        if epoch != 0:
          ave = total_time / epoch
          remaining = (args.max_epoches - epoch) * ave
        else: remaining = 0.0

        days = int(remaining / 86400)
        remaining %= 86400

        hours = int(remaining / 3600)
        remaining %= 3600

        minutes = int(remaining / 60)
        sec = remaining % 60
        logger.info('%d. loss: %f | training : %f | validation: %f' %
          (epoch, loss, train_accuracy, valid_accuarcy))
        logger.info('average: %f, time remaining: %d days %d:%d:%d' %
          (ave, days, hours, minutes, sec))

      start = time.time()
      sess.run(recognizer.train_ops, feed_dict={
        recognizer.inputs: data_batch,
        recognizer.labels: label_batch,
        recognizer.keep_prob: args.keep_prob,
      })
      passed = time.time() - start
      total_time += passed

      if epoch % args.save_epoch == 0 and args.saving and epoch != 0:
        saver.save(sess, checkpoint, global_step=epoch)
        summary = sess.run(recognizer.summary,
          feed_dict={
            recognizer.inputs: data_batch,
            recognizer.labels: label_batch,
            recognizer.validation_inputs: validation_data[o:o+batch_size, :],
            recognizer.validation_labels: validation_labels[o:o+batch_size, :],
            recognizer.keep_prob: 1.0,
          })
        summary_writer.add_summary(summary, global_step=epoch)

      if epoch % args.decay_epoch == 0 and epoch != 0:
        sess.run(recognizer.decay_lr)


def checkpoint_valid(checkpoint):
  return os.path.isfile(checkpoint + '.meta') and \
      os.path.isfile(checkpoint + '.index')


def recognize(dbname, args):
  loader = PlantLoader(dbname)
  loader.load_data()

  recognizer = PlantRecognizer(args.inference, 0,
    loader.get_width(), loader.get_height(),
    loader.get_output_size())

  if hasattr(args, 'checkpoint') and checkpoint_valid(args.checkpoint):
    logger.info('loading %s...' % args.checkpoint)
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
      logger.info('restoring %s...' % (args.checkpoint))
      saver.restore(sess, args.checkpoint)

      label_name = loader.get_label_name()
      test_images = loader.get_test_images()
      test_files = loader.get_test_files()
      with open(args.output_file, 'w') as f:
        f.write('file,species\n')

        for i in range(len(test_images)):
          img = test_images[i:i+1]
          filename = test_files[i]

          output = sess.run(recognizer.outputs, feed_dict={
            recognizer.inputs: img,
            recognizer.keep_prob: 1.0
          })
          result = label_name[np.argmax(output, axis=1)[0]][0]
          f.write('%s,%s\n' % (filename, result))
  else:
    logger.warning('%s not valid checkpoint' % args.checkpoint)


def main():
  parser = ArgumentParser()
  parser.add_argument('--dbname', dest='dbname', default='plants.sqlite3',
    type=str, help='db to load')
  parser.add_argument('--mode', dest='mode', default='train',
    type=str, help='train/test')

  parser.add_argument('--load-all', dest='load_all', default=False,
    type=bool, help='load all data to train')

  parser.add_argument('--learning-rate', dest='learning_rate', default=1e-4,
    type=float, help='learning rate to train model')
  parser.add_argument('--max-epoches', dest='max_epoches', default=60000,
    type=int, help='max epoches to train model')
  parser.add_argument('--display-epoches', dest='display_epoch', default=50,
    type=int, help='epoches to evaluation')
  parser.add_argument('--save-epoches', dest='save_epoch', default=1000,
    type=int, help='epoches to save model')
  parser.add_argument('--batch-size', dest='batch_size', default=64,
    type=int, help='batch size to train model')
  parser.add_argument('--saving', dest='saving', default=False,
    type=bool, help='rather to save model or not')
  parser.add_argument('--keep-prob', dest='keep_prob', default=0.8,
    type=float, help='keep probability for dropout')
  parser.add_argument('--decay-epoch', dest='decay_epoch', default=10000,
    type=int, help='epoches to decay learning rate')

  parser.add_argument('--checkpoint', dest='checkpoint',
    type=str, help='checkpoint to load for recognition')
  parser.add_argument('--output-file', dest='output_file', default='output.csv')
  parser.add_argument('--inference', dest='inference', default='inference_v2',
    type=str, help='inference function name')
  args = parser.parse_args()

  if args.mode == 'train':
    train(args.dbname, args)
  elif args.mode == 'test':
    recognize(args.dbname, args)


if __name__ == '__main__':
  main()

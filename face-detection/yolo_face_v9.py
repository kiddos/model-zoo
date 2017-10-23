from __future__ import print_function

import tensorflow as tf
import numpy as np
import os


IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
IMAGE_CHANNEL = 3
GRID_SIZE = 13


class YOLOFace(object):
  def __init__(self, learning_rate=1e-3,
      lambda_coord=5.0, lambda_noobj=0.5,
      lambda_indicator=1.0):
    self.lambda_coord = lambda_coord
    self.lambda_noobj = lambda_noobj
    self.lambda_indicator = lambda_indicator
    self._build_model(learning_rate)

  def _build_model(self, learning_rate):
    with tf.name_scope('inputs'):
      with tf.device('/cpu:0'):
        self.images = tf.placeholder(dtype=tf.float32,
          shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL],
          name='input_images')
        self.indicator_label = tf.placeholder(dtype=tf.float32,
          shape=[None, GRID_SIZE, GRID_SIZE, 1], name='indicator_label')
        self.coordinate_label = tf.placeholder(dtype=tf.float32,
          shape=[None, GRID_SIZE, GRID_SIZE, 2], name='coordinate_label')
        self.size_label = tf.placeholder(dtype=tf.float32,
          shape=[None, GRID_SIZE, GRID_SIZE, 2], name='size_label')

        tf.summary.image(name='input_images', tensor=self.images)
        tf.summary.image(name='indicator_label', tensor=self.indicator_label)

        self.keep_prob = tf.placeholder(dtype=tf.float32,
          shape=[], name='keep_prob')
        self.learning_rate = tf.Variable(learning_rate, trainable=False,
          name='learning_rate')
        self.decay_lr = tf.assign(self.learning_rate,
          self.learning_rate * 0.9, name='decay_lr')

        tf.summary.scalar(name='learning_rate', tensor=self.learning_rate)

    with tf.name_scope('conv1'):
      conv1_size = 16
      w = tf.get_variable(name='conv1_w',
        shape=[3, 3, IMAGE_CHANNEL, conv1_size],
        initializer=tf.random_normal_initializer(stddev=0.003))
      b = tf.get_variable(name='conv1_b', shape=[conv1_size],
        initializer=tf.constant_initializer(value=1e-3))
      conv1 = tf.nn.relu(tf.nn.conv2d(self.images, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('pool1'):
      pool1 = tf.nn.max_pool(conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    with tf.name_scope('conv2'):
      conv2_size = 32
      w = tf.get_variable(name='conv2_w',
        shape=[3, 3, conv1_size, conv2_size],
        initializer=tf.random_normal_initializer(stddev=0.1))
      b = tf.get_variable(name='conv2_b', shape=[conv2_size],
        initializer=tf.constant_initializer(value=1e-3))
      conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('pool2'):
      pool2 = tf.nn.max_pool(conv2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    with tf.name_scope('conv3'):
      conv3_size = 64
      w = tf.get_variable(name='conv3_w',
        shape=[3, 3, conv2_size, conv3_size],
        initializer=tf.random_normal_initializer(stddev=0.08))
      b = tf.get_variable(name='conv3_b', shape=[conv3_size],
        initializer=tf.constant_initializer(value=1e-3))
      conv3 = tf.nn.relu(tf.nn.conv2d(pool2, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('pool3'):
      pool3 = tf.nn.max_pool(conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    with tf.name_scope('conv4'):
      conv4_size = 128
      w = tf.get_variable(name='conv4_w',
        shape=[3, 3, conv3_size, conv4_size],
        initializer=tf.random_normal_initializer(stddev=0.06))
      b = tf.get_variable(name='conv4_b', shape=[conv4_size],
        initializer=tf.constant_initializer(value=1e-3))
      conv4 = tf.nn.relu(tf.nn.conv2d(pool3, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('pool4'):
      pool4 = tf.nn.max_pool(conv4, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    with tf.name_scope('conv5'):
      conv5_size = 256
      w = tf.get_variable(name='conv5_w',
        shape=[3, 3, conv4_size, conv5_size],
        initializer=tf.random_normal_initializer(stddev=0.03))
      b = tf.get_variable(name='conv5_b', shape=[conv5_size],
        initializer=tf.constant_initializer(value=1e-3))
      conv5 = tf.nn.relu(tf.nn.conv2d(pool4, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('pool5'):
      pool5 = tf.nn.max_pool(conv5, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    with tf.name_scope('conv6'):
      conv6_size = 512
      w = tf.get_variable(name='conv6_w',
        shape=[3, 3, conv5_size, conv6_size],
        initializer=tf.random_normal_initializer(stddev=0.02))
      b = tf.get_variable(name='conv6_b', shape=[conv6_size],
        initializer=tf.constant_initializer(value=1e-3))
      conv6 = tf.nn.relu(tf.nn.conv2d(pool5, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('pool6'):
      pool6 = tf.nn.max_pool(conv6, strides=[1, 1, 1, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    with tf.name_scope('conv7'):
      conv7_size = 1024
      w = tf.get_variable(name='conv7_w',
        shape=[3, 3, conv6_size, conv7_size],
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv7_b', shape=[conv7_size],
        initializer=tf.constant_initializer(value=1e-3))
      conv7 = tf.nn.relu(tf.nn.conv2d(pool6, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('conv8'):
      conv8_size = 1024
      w = tf.get_variable(name='conv8_w',
        shape=[3, 3, conv7_size, conv8_size],
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv8_b', shape=[conv8_size],
        initializer=tf.constant_initializer(value=1e-3))
      conv8 = tf.nn.relu(tf.nn.conv2d(conv7, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('fully_connected'):
      connect_shape = conv8.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      output_size = GRID_SIZE * GRID_SIZE * 5
      w = tf.get_variable(name='fcw',
        shape=[connect_size, output_size],
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0/connect_size)))
      b = tf.get_variable(name='fcb', shape=[output_size],
        initializer=tf.constant_initializer(value=1e-3))
      output = tf.matmul(tf.reshape(conv8, [-1, connect_size]), w) + b
      output = tf.reshape(output, [-1, GRID_SIZE, GRID_SIZE, 5])
      self.indicator_output, self.coord_output, self.size_output = \
        tf.split(output, [1, 2, 2], axis=3)
      tf.summary.image(name='indicator_output', tensor=self.indicator_output)

    with tf.name_scope('loss'):
      with tf.device('/cpu:0'):
        batch_size = tf.cast(tf.shape(self.images)[0], tf.float32)
      with tf.name_scope('indicator'):
        with tf.device('/cpu:0'):
          #  indicator_label = tf.reshape(self.indicator_label,
          #    [-1, GRID_SIZE * GRID_SIZE])
          noobj = 1.0 - self.indicator_label
        diff = self.indicator_output - self.indicator_label
        self.indicator_loss = tf.reduce_sum(
          tf.square(self.indicator_label * diff)) / batch_size
        self.noobj_loss = tf.reduce_sum(tf.square(noobj * diff)) / batch_size

      with tf.name_scope('coord'):
        diff = self.coord_output - self.coordinate_label
        self.coord_loss = tf.reduce_sum(
          tf.square(self.indicator_label * diff)) / batch_size

      with tf.name_scope('size'):
        diff = self.size_output - self.size_label
        self.size_loss = tf.reduce_sum(
          tf.square(self.indicator_label * diff)) / batch_size

      self.loss = \
        self.lambda_indicator * self.indicator_loss + \
        self.lambda_coord * self.coord_loss + \
        self.lambda_coord * self.size_loss + \
        self.lambda_noobj * self.noobj_loss

      with tf.device('/cpu:0'):
        tf.summary.scalar(name='total_loss', tensor=self.loss)
        tf.summary.scalar(name='indicator_loss', tensor=self.indicator_loss)
        tf.summary.scalar(name='noobj_loss', tensor=self.noobj_loss)
        tf.summary.scalar(name='coord_loss', tensor=self.coord_loss)
        tf.summary.scalar(name='size_loss', tensor=self.size_loss)

    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()


def test():
  YOLOFace()
  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
  test()

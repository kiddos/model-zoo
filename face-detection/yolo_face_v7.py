from __future__ import print_function

import tensorflow as tf
import numpy as np
import os


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 222
IMAGE_CHANNEL = 3
GRID_SIZE = 16


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
        initializer=tf.constant_initializer(value=1.0))
      conv1 = tf.nn.relu(tf.nn.conv2d(self.images, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      pool1 = tf.nn.max_pool(conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      tf.summary.histogram(name='pool1', values=pool1)

    with tf.name_scope('conv2'):
      conv2_size = 32
      w = tf.get_variable(name='conv2_w',
        shape=[3, 3, conv1_size, conv2_size],
        initializer=tf.random_normal_initializer(stddev=0.08))
      b = tf.get_variable(name='conv2_b', shape=[conv2_size],
        initializer=tf.constant_initializer(value=1.0))
      conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1],
        padding='SAME'))

    with tf.name_scope('conv3'):
      conv3_size = 32
      w = tf.get_variable(name='conv3_w',
        shape=[3, 3, conv2_size, conv3_size],
        initializer=tf.random_normal_initializer(stddev=0.08))
      b = tf.get_variable(name='conv3_b', shape=[conv3_size],
        initializer=tf.constant_initializer(value=1.0))
      conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w, strides=[1, 1, 1, 1],
        padding='SAME'))
      pool3 = tf.nn.max_pool(conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      tf.summary.histogram(name='pool3', values=pool3)

    with tf.name_scope('conv4'):
      conv4_size = 64
      w = tf.get_variable(name='conv4_w',
        shape=[3, 3, conv3_size, conv4_size],
        initializer=tf.random_normal_initializer(stddev=0.04))
      b = tf.get_variable(name='conv4_b', shape=[conv4_size],
        initializer=tf.constant_initializer(value=1.0))
      conv4 = tf.nn.relu(tf.nn.conv2d(pool3, w, strides=[1, 1, 1, 1],
        padding='SAME'))

    with tf.name_scope('conv5'):
      conv5_size = 64
      w = tf.get_variable(name='conv5_w',
        shape=[3, 3, conv4_size, conv5_size],
        initializer=tf.random_normal_initializer(stddev=0.04))
      b = tf.get_variable(name='conv5_b', shape=[conv5_size],
        initializer=tf.constant_initializer(value=1.0))
      conv5 = tf.nn.relu(tf.nn.conv2d(conv4, w, strides=[1, 1, 1, 1],
        padding='SAME'))
      pool5 = tf.nn.max_pool(conv5, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      tf.summary.histogram(name='pool5', values=pool5)

    with tf.name_scope('conv6'):
      conv6_size = 64
      w = tf.get_variable(name='conv6_w',
        shape=[3, 3, conv5_size, conv6_size],
        initializer=tf.random_normal_initializer(stddev=0.03))
      b = tf.get_variable(name='conv6_b', shape=[conv6_size],
        initializer=tf.constant_initializer(value=1.0))
      conv6 = tf.nn.relu(tf.nn.conv2d(pool5, w, strides=[1, 1, 1, 1],
        padding='SAME'))

    with tf.name_scope('conv7'):
      conv7_size = 128
      w = tf.get_variable(name='conv7_w',
        shape=[3, 3, conv6_size, conv7_size],
        initializer=tf.random_normal_initializer(stddev=0.03))
      b = tf.get_variable(name='conv7_b', shape=[conv7_size],
        initializer=tf.constant_initializer(value=1.0))
      conv7 = tf.nn.relu(tf.nn.conv2d(conv6, w, strides=[1, 1, 1, 1],
        padding='SAME'))
      pool7 = tf.nn.max_pool(conv7, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      tf.summary.histogram(name='pool7', values=pool7)

    with tf.name_scope('conv8'):
      conv8_size = 256
      w = tf.get_variable(name='conv8_w',
        shape=[3, 3, conv7_size, conv8_size],
        initializer=tf.random_normal_initializer(stddev=0.02))
      b = tf.get_variable(name='conv8_b', shape=[conv8_size],
        initializer=tf.constant_initializer(value=1.0))
      conv8 = tf.nn.relu(tf.nn.conv2d(pool7, w, strides=[1, 1, 1, 1],
        padding='SAME'))

    with tf.name_scope('conv9'):
      conv9_size = 256
      w = tf.get_variable(name='conv9_w',
        shape=[3, 3, conv8_size, conv9_size],
        initializer=tf.random_normal_initializer(stddev=0.02))
      b = tf.get_variable(name='conv9_b', shape=[conv9_size],
        initializer=tf.constant_initializer(value=1.0))
      conv9 = tf.nn.relu(tf.nn.conv2d(conv8, w, strides=[1, 1, 1, 1],
        padding='SAME'))
      tf.summary.histogram(name='conv16', values=conv9)

      with tf.device('/cpu:0'):
        drop9 = tf.nn.dropout(conv9, self.keep_prob)
    connect = drop9

    connect_shape = connect.get_shape().as_list()
    print('connect shape: %s' % (str(connect_shape)))
    connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
    with tf.name_scope('output'):
      with tf.name_scope('incicator'):
        w = tf.get_variable(name='indicator_w',
          shape=[connect_size, GRID_SIZE * GRID_SIZE],
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0/connect_size)))
        b = tf.get_variable(name='indicator_b',
          shape=[GRID_SIZE * GRID_SIZE],
          initializer=tf.constant_initializer(value=1.0))
        indicator_output = tf.matmul(
          tf.reshape(connect, [-1, connect_size]), w) + b
        with tf.device('/cpu:0'):
          self.indicator_output = tf.reshape(indicator_output,
            [-1, GRID_SIZE, GRID_SIZE, 1])
          tf.summary.image(name='indicator_prediction',
            tensor=self.indicator_output)

      with tf.name_scope('coordinate'):
        w = tf.get_variable(name='coord_w',
          shape=[connect_size, GRID_SIZE * GRID_SIZE * 2],
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0/connect_size)))
        b = tf.get_variable(name='coord_b',
          shape=[GRID_SIZE * GRID_SIZE * 2],
          initializer=tf.constant_initializer(value=1.0))
        coord_output = tf.matmul(
          tf.reshape(connect, [-1, connect_size]), w) + b
        with tf.device('/cpu:0'):
          self.coord_output = tf.reshape(coord_output,
            [-1, GRID_SIZE, GRID_SIZE, 2])

      with tf.name_scope('size'):
        w = tf.get_variable(name='size_w',
          shape=[connect_size, GRID_SIZE * GRID_SIZE * 2],
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0/connect_size)))
        b = tf.get_variable(name='size_b',
          shape=[GRID_SIZE * GRID_SIZE * 2],
          initializer=tf.constant_initializer(value=1.0))
        size_output = tf.matmul(
          tf.reshape(connect, [-1, connect_size]), w) + b
        with tf.device('/cpu:0'):
          self.size_output = tf.reshape(size_output,
            [-1, GRID_SIZE, GRID_SIZE, 2])

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

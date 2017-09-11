import tensorflow as tf
import numpy as np
import os


IMAGE_WIDTH = 128
IMAGE_HEIGHT = 111
IMAGE_CHANNEL = 3
GRID_SIZE = 7


class YOLOFace(object):
  def __init__(self, learning_rate=1e-3,
      lambda_coord=5.0, lambda_noobj=0.5):
    self.lambda_coord = lambda_coord
    self.lambda_noobj = lambda_noobj
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

        self.learning_rate = tf.Variable(learning_rate, trainable=False,
          name='learning_rate')

    with tf.name_scope('conv1'):
      conv1_size = 32
      w = tf.get_variable(name='conv1_w',
        shape=[3, 3, IMAGE_CHANNEL, conv1_size],
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv1_b', shape=[conv1_size],
        initializer=tf.constant_initializer(value=0.1))
      conv1 = tf.nn.relu(tf.nn.conv2d(self.images, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      pool1 = tf.nn.max_pool(conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    with tf.name_scope('conv2'):
      conv2_size = 64
      w = tf.get_variable(name='conv2_w',
        shape=[3, 3, conv1_size, conv2_size],
        initializer=tf.random_normal_initializer(stddev=0.05))
      b = tf.get_variable(name='conv2_b', shape=[conv2_size],
        initializer=tf.constant_initializer(value=0.1))
      conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1],
        padding='SAME'))
      pool2 = tf.nn.max_pool(conv2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    with tf.name_scope('conv3'):
      conv3_size = 128
      w = tf.get_variable(name='conv3_w',
        shape=[3, 3, conv2_size, conv3_size],
        initializer=tf.random_normal_initializer(stddev=0.05))
      b = tf.get_variable(name='conv3_b', shape=[conv3_size],
        initializer=tf.constant_initializer(value=0.1))
      conv3 = tf.nn.relu(tf.nn.conv2d(pool2, w, strides=[1, 1, 1, 1],
        padding='SAME'))
      pool3 = tf.nn.max_pool(conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')

    connect_shape = pool3.get_shape().as_list()
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
          tf.reshape(pool3, [-1, connect_size]), w) + b
        with tf.device('/cpu:0'):
          self.indicator_output = tf.reshape(indicator_output,
            [-1, GRID_SIZE, GRID_SIZE, 1])

      with tf.name_scope('coordinate'):
        w = tf.get_variable(name='coord_w',
          shape=[connect_size, GRID_SIZE * GRID_SIZE * 2],
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0/connect_size)))
        b = tf.get_variable(name='coord_b',
          shape=[GRID_SIZE * GRID_SIZE * 2],
          initializer=tf.constant_initializer(value=1.0))
        coord_output = tf.matmul(
          tf.reshape(pool3, [-1, connect_size]), w) + b
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
          tf.reshape(pool3, [-1, connect_size]), w) + b
        with tf.device('/cpu:0'):
          self.size_output = tf.reshape(size_output,
            [-1, GRID_SIZE, GRID_SIZE, 2])

    with tf.name_scope('loss'):
      with tf.name_scope('indicator'):
        with tf.device('/cpu:0'):
          indicator_label = tf.reshape(self.indicator_label,
            [-1, GRID_SIZE * GRID_SIZE])
          noobj = 1.0 - indicator_label
        diff = indicator_output - indicator_label
        self.indicator_loss = tf.reduce_mean(tf.square(indicator_label * diff))
        self.noobj_loss = tf.reduce_mean(tf.square(noobj * diff))

      with tf.name_scope('coord'):
        diff = self.coord_output - self.coordinate_label
        self.coord_loss = tf.reduce_mean(
          tf.square(self.indicator_label * diff))

      with tf.name_scope('size'):
        diff = self.size_output - self.size_label
        self.size_loss = tf.reduce_mean(
          tf.square(self.indicator_label * diff))

      self.loss = self.indicator_loss + \
        self.lambda_coord * self.coord_loss + \
        self.lambda_coord * self.size_loss + \
        self.lambda_noobj * self.noobj_loss

    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)


def test():
  YOLOFace()
  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
  test()

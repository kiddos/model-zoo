import tensorflow as tf
import numpy as np
import logging
import os


logging.basicConfig()
logger = logging.getLogger('mnist v2')
logger.setLevel(logging.INFO)


MODEL_NAME = 'MNISTV2'
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1
OUTPUT_SIZE = 10


class MNISTV2(object):
  def __init__(self, learning_rate=1e-3, decay=0.9):
    self._build_model(learning_rate, decay)

  def _build_model(self, learning_rate, decay):
    logger.info('setting up model...')
    with tf.device('/cpu:0'):
      self.images = tf.placeholder(
        shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL],
        dtype=tf.float32, name='input_images')
      self.labels = tf.placeholder(
        shape=[None, OUTPUT_SIZE],
        dtype=tf.float32, name='labels')
      self.keep_prob = tf.placeholder(shape=[],
        dtype=tf.float32, name='keep_prob')
      self.learning_rate = tf.Variable(learning_rate)
      self.decay_lr = tf.assign(self.learning_rate,
        self.learning_rate * decay)

    with tf.name_scope('conv1'):
      conv1_size = 256
      w = tf.get_variable(name='conv_w1',
        shape=[7, 7, IMAGE_CHANNEL, conv1_size],
        initializer=tf.random_normal_initializer(stddev=0.001))
      b = tf.get_variable(name='conv_b1',
        shape=[conv1_size], initializer=tf.constant_initializer(value=1e-1))
      conv1 = tf.nn.relu(tf.nn.conv2d(self.images, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('conv2'):
      conv2_size = 256
      w = tf.get_variable(name='conv_w2',
        shape=[5, 5, conv1_size, conv2_size],
        initializer=tf.random_normal_initializer(stddev=0.001))
      b = tf.get_variable(name='conv_b2',
        shape=[conv2_size], initializer=tf.constant_initializer(value=1e-1))
      conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('conv3'):
      conv3_size = 1024
      w = tf.get_variable(name='conv_w3',
        shape=[5, 5, conv2_size, conv3_size],
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b3',
        shape=[conv3_size], initializer=tf.constant_initializer(value=1e-1))
      conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      pool3 = tf.nn.max_pool(conv3, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1],
        padding='SAME')

    with tf.name_scope('conv4'):
      conv4_size = 1024
      w = tf.get_variable(name='conv_w4',
        shape=[5, 5, conv3_size, conv4_size],
        initializer=tf.random_normal_initializer(stddev=0.02))
      b = tf.get_variable(name='conv_b4',
        shape=[conv4_size], initializer=tf.constant_initializer(value=1e-1))
      conv4 = tf.nn.relu(tf.nn.conv2d(pool3, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      pool4 = tf.nn.max_pool(conv4, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1],
        padding='SAME')
      drop4 = tf.nn.dropout(pool4, self.keep_prob)

    with tf.name_scope('fc5'):
      conv4_shape = drop4.get_shape().as_list()
      connect_size = conv4_shape[1] * conv4_shape[2] * conv4_shape[3]
      fc5_size = 1024
      w = tf.get_variable(name='fc_w5', shape=[connect_size, fc5_size],
        initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 /
          connect_size)))
      b = tf.get_variable(name='fc_b5', shape=[fc5_size],
        initializer=tf.constant_initializer(value=1e-1))
      fc5 = tf.nn.relu(tf.matmul(tf.reshape(drop4, [-1, connect_size]), w) + b)

    with tf.name_scope('output'):
      w = tf.get_variable(name='output_w', shape=[fc5_size, OUTPUT_SIZE],
        initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 /
          fc5_size)))
      b = tf.get_variable(name='output_b', shape=[OUTPUT_SIZE],
        initializer=tf.constant_initializer(value=1e-1))
      logits = tf.matmul(fc5, w) + b
      self.outputs = tf.nn.softmax(logits)

    logger.info('setting up loss...')
    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,
          labels=self.labels))

    logger.info('setting up optimizer...')
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

    with tf.name_scope('accuracy'):
      self.accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(self.labels, axis=1),
          tf.argmax(self.outputs, axis=1)), tf.float32)) * 100.0


def test():
  model = MNISTV2()

  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    sess.run(model.train_ops, feed_dict={
      model.images: np.random.randn(100, IMAGE_SIZE, IMAGE_SIZE,IMAGE_CHANNEL),
      model.labels: np.random.randn(100, OUTPUT_SIZE),
      model.keep_prob: 1.0
    })


if __name__ == '__main__':
  test()

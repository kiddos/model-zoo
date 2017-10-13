import tensorflow as tf
import numpy as np
from argparse import ArgumentParser


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 222
IMAGE_CHANNEL = 3
DECISION_COUNT = 64


class LookTwice(object):
  def __init__(self, learning_rate=1e-3):
    self._build_model(learning_rate)

  def _build_model(self, learning_rate):
    with tf.device('/cpu:0'):
      self.learning_rate = tf.Variable(learning_rate,
        name='learning_rate', trainable=False)
    with tf.name_scope('input'):
      with tf.device('/cpu:0'):
        self.images = tf.placeholder(dtype=tf.float32,
          shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL],
          name='input_images')
        self.indicator_label = tf.placeholder(dtype=tf.float32,
          shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 1],
          name='indicator_label')
        self.size_label = tf.placeholder(dtype=tf.float32,
          shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 1],
          name='size_label')
        self.keep_prob = tf.placeholder(dtype=tf.float32,
          shape=(), name='keep_prob')

    with tf.name_scope('conv_small'):
      w = tf.get_variable(name='conv_small_w',
        shape=[7, 7, IMAGE_CHANNEL, DECISION_COUNT])
      b = tf.get_variable(name='conv_small_b',
        shape=[DECISION_COUNT])
      small = tf.nn.relu(tf.nn.conv2d(self.images, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('conv_medium'):
      w = tf.get_variable(name='conv_medium_w',
        shape=[15, 15, IMAGE_CHANNEL, DECISION_COUNT])
      b = tf.get_variable(name='conv_medium_b',
        shape=[DECISION_COUNT])
      medium = tf.nn.relu(tf.nn.conv2d(self.images, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('conv_large'):
      w = tf.get_variable(name='conv_large_w',
        shape=[15, 15, IMAGE_CHANNEL, DECISION_COUNT])
      b = tf.get_variable(name='conv_large_b',
        shape=[DECISION_COUNT])
      large = tf.nn.relu(tf.nn.conv2d(self.images, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)

    with tf.name_scope('hidden'):
      hidden = tf.concate([small, medium, large], axis=3)

    with tf.name_scope('conv2'):
      w = tf.get_variable(name='conv2_w',
        shape=[5, 5, IMAGE_CHANNEL, DECISION_COUNT * 3])
      b = tf.get_variable(name='conv2_b',
        shape=[DECISION_COUNT * 3])
      conv2 = tf.nn.relu(tf.nn.conv2d(hidden, w, strides=[1, 1, 1, 1]) + b)

    with tf.name_scope('conv3'):
      w = tf.get_variable(name='conv3_W',
        shape=[5, 5, IMAGE_CHANNEL, DECISION_COUNT * 3])
      b = tf.get_variable(name='conv3_b',
        shape=[DECISION_COUNT * 3])
      conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w, strides=[1, 1, 1, 1]) + b)

    with tf.name_scope('output'):
      output = conv3

    with tf.name_scope('loss'):
      loss = tf.reduce_mean(tf.square(output - self.indicator_label))


def main():
  parser = ArgumentParser()
  parser.add_argument('--learning-rate', dest='learning_rate',
    default=1e-3, type=float, help='learning rate to train')
  parser.add_argument('--display-epoch', dest='display_epoch',
    default=100, type=int, help='epoch to display')
  parser.add_argument('--save-epoch', dest='save_epoch',
    default=10000, type=int, help='epoch to save')


if __name__ == '__main__':
  main()

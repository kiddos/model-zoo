import tensorflow as tf
import unittest
import os
import numpy as np


class PlantNaiveModel(object):
  def __init__(self, input_size, num_classes,
      learning_rate, lambda_reg):
    self.input_size = input_size
    self.num_classes = num_classes
    self._set_inputs()

    with tf.variable_scope('inference'):
      self.logits, self.output = self._inference(self.images, self.keep_prob)

    with tf.variable_scope('inference', reuse=True):
      _, valid_output = self._inference(self.valid_images, 1)

    with tf.name_scope('loss'):
      var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inference')
      assert len(var) > 0
      reg_term = 0
      for i in range(len(var)):
        if len(var[i].get_shape().as_list()) > 1:
          reg_term += tf.reduce_sum(tf.square(var[i]))
      assert reg_term != 0

      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=self.logits, labels=self.labels))
      self.loss = tf.add(self.loss, reg_term * lambda_reg, name='loss')

      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimizer'):
      self.learning_rate = tf.Variable(learning_rate, trainable=False,
        name='learning_rate')
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

    with tf.name_scope('evaluation'):
      self.train_acc = self._evaluate(self.output, self.labels)
      tf.summary.scalar('training_accuracy', self.train_acc)

      self.valid_acc = self._evaluate(valid_output, self.valid_labels)
      tf.summary.scalar('validation_accuracy', self.valid_acc)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _set_inputs(self):
    with tf.name_scope('inputs'):
      s = self.input_size
      self.images = tf.placeholder(tf.float32, [None, s, s, 3], name='images')
      self.labels = tf.placeholder(tf.float32, [None, self.num_classes],
        name='label')
      self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

      self.valid_images = tf.placeholder(tf.float32, [None, s, s, 3],
        name='images')
      self.valid_labels = tf.placeholder(tf.float32, [None, self.num_classes],
        name='label')

      tf.summary.image('input_images', self.images, 1)
      tf.summary.image('valid_images', self.valid_images, 1)

  def _inference(self, inputs, keep_prob):
    init = tf.random_normal_initializer(stddev=0.001)
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, kernel_size=3, stride=1,
        weights_initializer=init)

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    init = tf.variance_scaling_initializer()
    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 64, kernel_size=3, stride=1,
        weights_initializer=init)

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop2'):
      drop = tf.nn.dropout(pool, keep_prob=keep_prob)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(drop, 128, kernel_size=3, stride=1,
        weights_initializer=init)

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop3'):
      drop = tf.nn.dropout(pool, keep_prob=keep_prob)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(drop, 256, kernel_size=3, stride=1,
        weights_initializer=init)

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop4'):
      drop = tf.nn.dropout(pool, keep_prob=keep_prob)

    with tf.name_scope('fully_connected'):
      flatten = tf.contrib.layers.flatten(drop)
      fc = tf.contrib.layers.fully_connected(flatten, 256,
        weights_initializer=init)

    with tf.name_scope('output'):
      logits = tf.contrib.layers.fully_connected(fc, self.num_classes,
        weights_initializer=init, activation_fn=None)
      outputs = tf.nn.softmax(logits, name='prediction')
    return logits, outputs

  def _evaluate(self, prediction, labels):
    correct = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(labels, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


class TestNaiveModel(unittest.TestCase):
  def setUp(self):
    self.model = PlantNaiveModel(64, 12, 1e-4, 1e-4)

  def test_output(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      output = sess.run(self.model.logits, feed_dict={
        self.model.images: np.random.randint(0, 255, [32, 64, 64, 3], np.uint8),
        self.model.keep_prob: 1.0,
      })
      print(np.std(output, axis=0))
      print(np.mean(output, axis=0))


if __name__ == '__main__':
  unittest.main()

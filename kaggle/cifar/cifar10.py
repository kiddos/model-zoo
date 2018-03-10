import tensorflow as tf
import numpy as np
import unittest


class CIFAR10Model(object):
  def __init__(self, learning_rate=1e-3, lambda_reg=1e-4):
    with tf.name_scope('inputs'):
      self._setup_inputs()
      tf.summary.image('input_images', self.input_images)

    with tf.variable_scope('cifar10'):
      self.logits, self.outputs = self.inference(self.input_images)

    with tf.variable_scope('cifar10', reuse=True):
      _, self.validation_output = self.inference(self.validation_images)

    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=self.logits, labels=self.labels))

      train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'cifar10')
      assert len(train_vars) != 0
      reg = 0
      for i in range(len(train_vars)):
        if len(train_vars[i].get_shape().as_list()) > 1:
          reg += tf.reduce_sum(tf.square(train_vars[i]))
      assert reg != 0

      self.loss += lambda_reg * reg
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(learning_rate, trainable=False,
        name='learning_rate')
      self.decay_lr = tf.assign(self.learning_rate,
        self.learning_rate * 0.9, name='decay_learning_rate')
      tf.summary.scalar('learning_rate', self.learning_rate)

      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

    with tf.name_scope('evaluation'):
      self.train_accuracy = self.evaluate(self.outputs, self.labels)
      self.validation_accuracy = self.evaluate(self.validation_output,
        self.validation_labels)
      tf.summary.scalar('train_accuarcy', self.train_accuracy)
      tf.summary.scalar('validation_accuracy', self.validation_accuracy)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _setup_inputs(self):
    self.input_images = tf.placeholder(dtype=tf.float32, name='input_images',
      shape=[None, 32, 32, 3])
    self.labels = tf.placeholder(dtype=tf.float32, name='labels',
      shape=[None, 10])

    self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob',
      shape=())

    self.validation_images = tf.placeholder(dtype=tf.float32,
      name='validation_images', shape=[None, 32, 32, 3])
    self.validation_labels = tf.placeholder(dtype=tf.float32,
      name='validation_labels', shape=[None, 10])

  def inference(self, inputs):
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 64, stride=1, kernel_size=3,
        weights_initializer=tf.random_normal_initializer(stddev=0.006))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop1'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('conv2'):
      conv = self.multiple_conv(drop, 128, multiples=0)

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop2'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('conv3'):
      conv = self.multiple_conv(drop, 256, multiples=0)

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop3'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('conv4'):
      conv = self.multiple_conv(drop, 512, multiples=1)

    with tf.name_scope('drop4'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    init = tf.variance_scaling_initializer()
    with tf.name_scope('fully_connected'):
      flatten = tf.contrib.layers.flatten(drop)
      fc = tf.contrib.layers.fully_connected(flatten, 1024,
        weights_initializer=init)

    with tf.name_scope('output'):
      logits = tf.contrib.layers.fully_connected(fc, 10,
        activation_fn=None, weights_initializer=init)
      output = tf.nn.softmax(logits, name='prediction')
    return logits, output

  def multiple_conv(self, inputs, output_size, ksize=3, multiples=1):
    init = tf.variance_scaling_initializer()
    conv = tf.contrib.layers.conv2d(inputs, output_size,
      stride=1, kernel_size=ksize, weights_initializer=init)

    for i in range(multiples):
      conv = tf.contrib.layers.conv2d(conv, output_size / 2,
        stride=1, kernel_size=1, weights_initializer=init)
      conv = tf.contrib.layers.conv2d(conv, output_size,
        stride=1, kernel_size=ksize, weights_initializer=init)
    return conv

  def evaluate(self, prediction, labels):
    p = tf.argmax(prediction, axis=1)
    l = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(p, l), tf.float32))


class TestCIFAR10Model(unittest.TestCase):
  def setUp(self):
    self.model = CIFAR10Model(1e-4, 1e-4)

  def test_run(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      logits = sess.run(self.model.logits, feed_dict={
        self.model.input_images: np.random.randint(0, 255, [32, 32, 32, 3]),
        self.model.keep_prob: 1.0
      })
      print(np.mean(logits, axis=0))
      print(np.std(logits, axis=0))


if __name__ == '__main__':
  unittest.main()

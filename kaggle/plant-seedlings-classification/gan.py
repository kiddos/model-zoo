import tensorflow as tf
import numpy as np
import unittest


def conv2d(inputs, output_size, ksize, strides, padding, initializer, name,
    activation=None, const=False):
  input_channel = inputs.get_shape().as_list()[-1]
  w = tf.get_variable('%s_w' % (name),
    [ksize, ksize, input_channel, output_size], initializer=initializer)
  b = tf.get_variable('%s_b' % (name), [output_size],
    initializer=tf.zeros_initializer())

  if const:
    w = tf.stop_gradient(w)
    b = tf.stop_gradient(b)

  conv = tf.nn.conv2d(inputs, w, strides, padding=padding) + b
  if activation:
    conv = activation(conv)
  return conv


def fully_connect(inputs, output_size, initializer, name, activation, const):
  input_size = inputs.get_shape().as_list()[-1]

  w = tf.get_variable('%s_w' % (name),
    [input_size, output_size], initializer=initializer)
  b = tf.get_variable('%s_b' % (name), [output_size],
    initializer=tf.zeros_initializer())

  if const:
    w = tf.stop_gradient(w)
    b = tf.stop_gradient(b)

  fc = tf.matmul(inputs, w) + b
  if activation:
    fc = activation(fc)
  return fc


class GAN(object):
  def __init__(self, input_shape, num_classes, learning_rate, lambda_reg):
    self.input_shape = input_shape
    self.num_classes = num_classes
    self.learning_rate = learning_rate

    assert self.input_shape[0] == self.input_shape[1]

    self._set_inputs()

    self._build_graph()

  def _set_inputs(self):
    with tf.name_scope('inputs'):
      self.input_images = tf.placeholder(tf.float32,
        [None] + self.input_shape, name='input_images')
      self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
      self.labels = tf.placeholder(tf.float32, [None, self.num_classes],
        name='labels')
      self.noise = tf.placeholder(tf.float32,
        self.input_shape, name='generator_noise')

  def _generator_inference(self, labels, noise):
    with tf.name_scope('reshape'):
      reshaped = tf.reshape(labels, [-1, 1, 1, self.num_classes])

    init = tf.contrib.layers.xavier_initializer()
    with tf.name_scope('conv1'):
      size = self.input_shape[0]
      conv = tf.contrib.layers.conv2d(reshaped, size ** 2,
        kernel_size=1, stride=1, weights_initializer=init)
      reshaped = tf.reshape(conv, [-1, size, size, 1])

    with tf.name_scope('conv2'):
      nw = tf.get_variable('nw', [3, 3, self.input_shape[-1], 64],
        initializer=init)
      cw = tf.get_variable('cw', [3, 3, 1, 64], initializer=init)
      b = tf.get_variable('b', [64],
        initializer=tf.zeros_initializer())
      noise = tf.expand_dims(noise, axis=0)
      conv = tf.nn.relu(
        tf.nn.conv2d(reshaped, cw, strides=[1, 1, 1, 1], padding='SAME') +
        tf.nn.conv2d(noise, nw, strides=[1, 1, 1, 1], padding='SAME') + b)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(conv, 128, kernel_size=3, stride=1,
        weights_initializer=init)

    with tf.name_scope('generated_images'):
      channel = self.input_shape[-1]
      ow = tf.get_variable('ow', [3, 3, 128, channel], initializer=init)
      ob = tf.get_variable('ob', [channel], initializer=tf.zeros_initializer())
      output = tf.add(
        tf.nn.conv2d(conv, ow, strides=[1, 1, 1, 1], padding='SAME'), ob,
        name='output')
    return output

  def _discriminator_inference(self, inputs, const=False):
    init = tf.variance_scaling_initializer()
    with tf.name_scope('conv1'):
      conv = conv2d(inputs, 64, 3, [1, 1, 1, 1], 'SAME', init, 'conv1',
        tf.nn.relu, const)

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop1'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('conv2'):
      conv = conv2d(inputs, 128, 3, [1, 1, 1, 1], 'SAME', init, 'conv2',
        tf.nn.relu, const)

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop2'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('conv3'):
      conv = conv2d(inputs, 256, 3, [1, 1, 1, 1], 'SAME', init, 'conv3',
        tf.nn.relu, const)

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop2'):
      drop = tf.nn.dropout(pool, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      flatten = tf.contrib.layers.flatten(drop)
      fc = fully_connect(flatten, 256, init, 'fc', tf.nn.relu, const)

    with tf.name_scope('outputs'):
      logits = fully_connect(fc, self.num_classes, init, 'outputs', None, const)
      outputs = tf.nn.sigmoid(logits, name='discriminator_outputs')
    return logits, outputs

  def _build_graph(self):
    with tf.variable_scope('generator'):
      self.generated_images = self._generator_inference(self.labels, self.noise)

    with tf.name_scope('norm'):
      true_inputs = tf.divide(self.input_images - 127, 128.0, name='norm_images')

    with tf.name_scope('process_inputs'):
      gimage = tf.stop_gradient(self.generated_images)
      fake_labels = tf.zeros_like(self.labels)
      all_inputs = tf.concat([true_inputs, gimage], axis=0)
      all_labels = tf.concat([self.labels, fake_labels], axis=0)

    with tf.variable_scope('discriminator'):
      self.dlogits, self.doutput = self._discriminator_inference(all_inputs)

    with tf.variable_scope('discriminator', reuse=True):
      self.glogits, self.goutput = \
        self._discriminator_inference(self.generated_images, True)

    with tf.name_scope('discrinimator_loss'):
      self.dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.dlogits, labels=all_labels), name='discriminator_loss')

    with tf.name_scope('generator_loss'):
      self.gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.glogits, labels=self.labels), name='generator_loss')

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(self.learning_rate, trainable=False,
        name='learning_rate')
      doptimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_discrinimator = doptimizer.minimize(self.dloss)

      goptimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_generator = goptimizer.minimize(self.gloss)


class TestGAN(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.gan = GAN([64, 64, 3], 12, 1e-4, 1e-4)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    cls.sess = tf.Session(config=config)
    cls.sess.run(tf.global_variables_initializer())

    cls.batch_size = 4
    cls.random_label = np.zeros([cls.batch_size, 12])
    cls.random_label[np.arange(cls.batch_size),
      np.random.randint(0, 12, [cls.batch_size])] = 1

  @classmethod
  def tearDownClass(cls):
    cls.sess.close()

  def __del__(self):
    self.sess.close()

  def test_run(self):
    generated_images = self.sess.run(self.gan.generated_images, feed_dict={
      self.gan.labels: self.random_label,
      self.gan.noise: np.random.randn(64, 64, 3),
    })
    gen = np.reshape(generated_images, [-1, 3])
    print(np.mean(gen, axis=0))
    print(np.std(gen, axis=0))

    dlogits = self.sess.run(self.gan.dlogits, feed_dict={
      self.gan.input_images: np.random.randint(0, 255,
        [self.batch_size, 64, 64, 3]),
      self.gan.labels: self.random_label,
      self.gan.noise: np.random.randn(64, 64, 3),
      self.gan.keep_prob: 1.0,
    })
    print(np.mean(dlogits, axis=0))
    print(np.std(dlogits, axis=0))

  def test_train_discriminator(self):
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'generator')

    g1 = self.sess.run(g_vars)
    self.sess.run(self.gan.train_discrinimator, feed_dict={
      self.gan.input_images: np.random.randint(0, 255,
        [self.batch_size, 64, 64, 3]),
      self.gan.labels: self.random_label,
      self.gan.noise: np.random.randn(64, 64, 3),
      self.gan.keep_prob: 1.0,
    })
    g2 = self.sess.run(g_vars)
    for i in range(len(g1)):
      eq = g1[i] == g2[i]
      self.assertTrue(eq.all())

  def test_train_generator(self):
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'discriminator')

    d1 = self.sess.run(d_vars)
    self.sess.run(self.gan.train_generator, feed_dict={
      self.gan.labels: self.random_label,
      self.gan.noise: np.random.randn(64, 64, 3),
      self.gan.keep_prob: 1.0,
    })
    d2 = self.sess.run(d_vars)

    for i in range(len(d2)):
      eq = d1[i] == d2[i]
      self.assertTrue(eq.all())


if __name__ == '__main__':
  unittest.main()

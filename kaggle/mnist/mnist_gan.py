import tensorflow as tf
import numpy as np
import unittest


class GAN(object):
  def __init__(self, learning_rate):
    self._setup_inputs()
    tf.summary.image('target_images', self.target_images)

    target_labels = tf.one_hot(self.target_labels, 10)
    with tf.variable_scope('generator'):
      self.mnist_generated = \
        self._generator(self.input_noise, target_labels, training=False)
      tf.summary.image('generated_images', self.mnist_generated)

    with tf.variable_scope('generator', reuse=True):
      self.generated = \
        self._generator(self.input_noise, target_labels)

    with tf.variable_scope('discriminator'):
      self.logits, self.outputs = \
        self._discriminator(self.target_images, training=False)

    with tf.variable_scope('discriminator', reuse=True):
      true_logits, _ = self._discriminator(self.target_images)

    with tf.variable_scope('discriminator', reuse=True):
      false_logits, _ = self._discriminator(self.generated)

    with tf.name_scope('loss'):
      self.d_loss1 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=true_logits,
          labels=target_labels))
      self.d_loss2 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=false_logits,
          labels=tf.zeros_like(target_labels)))
      self.d_loss = tf.add(self.d_loss1, self.d_loss2, name='d_loss')

      self.g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=false_logits,
          labels=target_labels), name='g_loss')

    with tf.name_scope('optimization'):
      g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'generator')
      d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'discriminator')

      self.learning_rate = tf.Variable(learning_rate, trainable=False,
        name='learning_rate')
      self.train_g = tf.train.AdamOptimizer(
        self.learning_rate).minimize(self.g_loss, var_list=g_vars)

      self.train_d = tf.train.AdamOptimizer(
        self.learning_rate).minimize(self.g_loss, var_list=d_vars)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _setup_inputs(self):
    self.target_images = tf.placeholder(dtype=tf.float32,
      shape=[None, 28, 28, 1], name='target_images')
    self.keep_prob = tf.placeholder(dtype=tf.float32,
      shape=(), name='keep_prob')
    self.target_labels = tf.placeholder(dtype=tf.int32,
      shape=[None], name='target_labels')
    self.input_noise = tf.placeholder(dtype=tf.float32,
      shape=[None, 28, 28, 1], name='input_noise')

  def _deconv(self, inputs, output_size, kernel_size, stride,
      leak=0.1, training=True):
    deconv = tf.layers.conv2d_transpose(inputs, output_size,
      kernel_size, strides=stride,
      activation=lambda x: tf.nn.leaky_relu(x, leak))
    if training:
      deconv = tf.nn.dropout(deconv, keep_prob=self.keep_prob)
    return deconv

  def _block(self, inputs, output_size, leak=0.1, training=True):
    init = tf.variance_scaling_initializer()

    conv = tf.layers.conv2d(inputs, output_size, 3,
      activation=lambda x: tf.nn.leaky_relu(x, leak), padding='SAME',
      kernel_initializer=init)
    conv = tf.layers.conv2d(conv, output_size, 1,
      activation=lambda x: tf.nn.leaky_relu(x, leak), padding='SAME',
      kernel_initializer=init)

    if training:
      conv = tf.nn.dropout(conv, keep_prob=self.keep_prob)
    return conv

  def _stack(self, conv, output_size, leak=0.1, training=True, multiple=1):
    for _ in range(multiple):
      conv = self._block(conv, output_size, leak, training=training)
    return conv

  def _down_sample(self, inputs, output_size, kernel_size,
      leak=0.1, training=True):
    init = tf.variance_scaling_initializer()
    conv = tf.layers.conv2d(inputs, output_size, kernel_size, strides=2,
      activation=lambda x: tf.nn.leaky_relu(x, leak), padding='SAME',
      kernel_initializer=init)

    if training:
      conv = tf.nn.dropout(conv, keep_prob=self.keep_prob)
    return conv

  def _generator(self, input_noise, label, training=True):
    with tf.name_scope('reshape_label'):
      label_input = tf.reshape(label, [-1, 1, 1, 10])

    with tf.name_scope('deconv1'):
      deconv = self._deconv(label_input, 16, 8, 1, training=training)

    with tf.name_scope('deconv2'):
      deconv = self._deconv(deconv, 32, 8, 2, training=training)

    with tf.name_scope('deconv3'):
      deconv = self._deconv(deconv, 64, 5, 1, training=training)

    with tf.name_scope('deconv4'):
      deconv = self._deconv(deconv, 128, 3, 1, training=training)

    init = tf.variance_scaling_initializer()
    with tf.name_scope('conv5'):
      conv1 = tf.layers.conv2d(deconv, 128, 3,
        activation=lambda x: tf.nn.leaky_relu(x, 0.1), padding='SAME',
        kernel_initializer=init)
      conv2 = tf.layers.conv2d(input_noise, 128, 3,
        activation=lambda x: tf.nn.leaky_relu(x, 0.1), padding='SAME',
        kernel_initializer=init)
      conv = tf.concat([conv1, conv2], axis=3)
      conv = self._block(conv, 256, training=training)

    with tf.name_scope('conv6'):
      conv = self._block(conv, 256, training=training)

    with tf.name_scope('output'):
      ow = tf.get_variable('ow', shape=[3, 3, 256, 1],
        initializer=init)
      ob = tf.get_variable('ob', shape=[1],
        initializer=tf.zeros_initializer())
      logits = tf.add(
        tf.nn.conv2d(conv, ow, strides=[1, 1, 1, 1], padding='SAME'), ob,
        name='logits')
    return logits

  def _discriminator(self, input_images, training=True):
    with tf.name_scope('conv1'):
      init = tf.random_normal_initializer(stddev=0.01)
      conv = tf.layers.conv2d(input_images, 32, 3,
        activation=lambda x: tf.nn.leaky_relu(x, 0.1), padding='SAME',
        kernel_initializer=init)

    with tf.name_scope('conv2'):
      conv = self._block(conv, 64, training=training)
      conv = self._down_sample(conv, 64, 3, training=training)

    with tf.name_scope('conv3'):
      conv = self._stack(conv, 128, training=training, multiple=2)
      conv = self._down_sample(conv, 128, 3, training=training)

    with tf.name_scope('conv4'):
      conv = self._stack(conv, 256, training=training, multiple=2)

    with tf.name_scope('conv5'):
      conv = self._stack(conv, 512, training=training, multiple=3)

    with tf.name_scope('conv6'):
      conv = self._block(conv, 1024, training=training)

    with tf.name_scope('output'):
      init = tf.variance_scaling_initializer()
      ow = tf.get_variable('ow', shape=[7, 7, 1024, 10],
        initializer=init)
      ob = tf.get_variable('ob', shape=[10],
        initializer=tf.zeros_initializer())
      logits = tf.nn.conv2d(conv, ow, strides=[1, 1, 1, 1], padding='VALID') + ob
      logits = tf.reshape(logits, [-1, 10], name='logits')
      outputs = tf.nn.sigmoid(logits)
    return logits, outputs


class TestGAN(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.gan = GAN(1e-3)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    cls.sess = tf.Session(config=config)
    cls.sess.run(tf.global_variables_initializer())

  @classmethod
  def tearDownClass(cls):
    cls.sess.close()

  def test_generator_output(self):
    labels = np.random.randint(0, 9, [1])
    noise = np.random.randn(1, 28, 28, 1)

    gen = self.sess.run(self.gan.mnist_generated, feed_dict={
      self.gan.target_labels: labels,
      self.gan.input_noise: noise,
    })
    print(np.mean(gen))
    print(np.std(gen))

  def test_discriminator_outptt(self):
    images = np.random.randint(0, 255, [1, 28, 28, 1])

    logits, outputs = self.sess.run([self.gan.logits, self.gan.outputs],
      feed_dict={
        self.gan.target_images: images,
      })
    print(np.mean(logits))
    print(np.std(logits))

    print(np.mean(outputs))
    print(np.std(outputs))


if __name__ == '__main__':
  unittest.main()

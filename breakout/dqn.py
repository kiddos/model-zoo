import tensorflow as tf
import numpy as np
import unittest


class DQNConfig(object):
  learning_rate = 0.00025
  gamma = 0.99
  decay = 0.95
  momentum = 0.95
  eps = 0.01
  input_width = 84
  input_height = 84
  skip = 4
  action_size = 4


class DQN(object):
  def __init__(self, config, use_huber):
    self.config = config
    self._setup_inputs()

    with tf.variable_scope('train'):
      self.q_values = self.inference(self.state)
    tf.summary.histogram('q_values', self.q_values)

    with tf.variable_scope('target'):
      self.next_q_values = self.inference(self.next_state, False)
    tf.summary.histogram('next_q_values', self.next_q_values)

    with tf.name_scope('copy_ops'):
      train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'train')
      target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target')
      self.copy_ops = []
      assert len(train_vars) == len(target_vars)
      assert len(train_vars) > 0
      for i in range(len(train_vars)):
        assert target_vars[i].get_shape().as_list() == \
            train_vars[i].get_shape().as_list()
        self.copy_ops.append(tf.assign(target_vars[i], train_vars[i]))
        assert target_vars[i] not in tf.trainable_variables()
        assert train_vars[i] in tf.trainable_variables()

    with tf.name_scope('loss'):
      self.target = tf.clip_by_value(self.reward, -1, 1) + \
        config.gamma * \
        tf.cast(tf.logical_not(self.done), tf.float32) * \
        tf.stop_gradient(tf.reduce_max(self.next_q_values, axis=1))

      action_mask = tf.one_hot(self.action,
        config.action_size, name='action_mask')
      self.y = tf.reduce_sum(action_mask * self.q_values, axis=1, name='y')

      self.loss = tf.losses.huber_loss(self.target, self.y,
        reduction=tf.losses.Reduction.MEAN)

      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(config.learning_rate,
        trainable=False, name='learning_rate')
      optimizer = tf.train.RMSPropOptimizer(
        self.learning_rate,
        decay=config.decay,
        momentum=config.momentum,
        epsilon=config.eps)
      #  optimizer = tf.train.AdamOptimizer(self.learning_rate,
      #    epsilon=1e-3)
      #  grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_vars), 10.0)
      #  self.train_ops = optimizer.apply_gradients(zip(grads, train_vars))
      self.train_ops = optimizer.minimize(self.loss)

      tf.summary.scalar('learning_rate', self.learning_rate)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _setup_inputs(self):
    c = self.config
    self.state = tf.placeholder(dtype=tf.uint8, name='state',
      shape=[None, c.input_height, c.input_width, c.skip])
    self.next_state = tf.placeholder(dtype=tf.uint8, name='next_state',
      shape=[None, c.input_height, c.input_width, c.skip])
    self.reward = tf.placeholder(dtype=tf.float32, name='reward', shape=[None])
    self.done = tf.placeholder(dtype=tf.bool, name='done', shape=[None])
    self.action = tf.placeholder(dtype=tf.int32, name='action',
      shape=[None])

  def inference(self, inputs, trainable=True):
    with tf.name_scope('norm'):
      inputs = tf.div(tf.cast(inputs, tf.float32), 255.0)
      images = tf.split(inputs, [1, 1, 1, 1], axis=3)
      for i in range(4):
        tf.summary.image('input_images_%d' % i, images[i])

    def act(inputs):
      return tf.nn.leaky_relu(inputs, alpha=0.01)

    initializer = tf.variance_scaling_initializer()
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=4, kernel_size=8,
        activation_fn=act, trainable=trainable, padding='SAME',
        weights_initializer=initializer)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(conv, 64, stride=2, kernel_size=4,
        activation_fn=act, trainable=trainable, padding='SAME',
        weights_initializer=initializer)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(conv, 64, stride=1, kernel_size=3,
        activation_fn=act, trainable=trainable, padding='SAME',
        weights_initializer=initializer)

    with tf.name_scope('fully_connected'):
      flatten = tf.contrib.layers.flatten(conv)
      fc = tf.contrib.layers.fully_connected(flatten, 512, trainable=trainable,
        activation_fn=act, weights_initializer=initializer)

    with tf.name_scope('output'):
      w = tf.get_variable('ow', shape=[512, self.config.action_size],
        trainable=trainable,
        initializer=initializer)
      b = tf.get_variable('ob', shape=[self.config.action_size],
        trainable=trainable,
        initializer=tf.zeros_initializer())
      outputs = tf.add(tf.matmul(fc, w), b, name='q_values')
    return outputs


class TestDQN(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    config = DQNConfig()
    cls.dqn = DQN(config, False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    cls.sess = tf.Session(config=config)

    cls.sess.run(tf.global_variables_initializer())

  @classmethod
  def tearDownClass(cls):
    cls.sess.close()

  def test_run(self):
    state = np.random.uniform(0, 255, [1, 84, 84, 4])
    q, nq = self.sess.run([self.dqn.q_values, self.dqn.next_q_values],
      feed_dict={
        self.dqn.state: state,
        self.dqn.next_state: state,
      })
    self.assertFalse((q == nq).any())
    self.sess.run(self.dqn.copy_ops)

    state = np.random.uniform(0, 255, [1, 84, 84, 4])
    q, nq = self.sess.run([self.dqn.q_values, self.dqn.next_q_values],
      feed_dict={
        self.dqn.state: state,
        self.dqn.next_state: state,
      })
    self.assertTrue((q == nq).all())
    self.sess.run(self.dqn.copy_ops)


  def test_y(self):
    state = np.random.uniform(0, 255, [1, 84, 84, 4])
    q = self.sess.run(self.dqn.q_values, feed_dict={
      self.dqn.state: state,
    })[0, :]

    y = self.sess.run(self.dqn.y, feed_dict={
      self.dqn.state: state,
      self.dqn.action: np.array([1]),
    })[0]
    self.assertEqual(y, q[1])

  def test_target(self):
    next_state = np.random.uniform(0, 255, [1, 84, 84, 4])
    next_q = self.sess.run(self.dqn.next_q_values, feed_dict={
      self.dqn.next_state: next_state,
    })[0, :]

    target = self.sess.run(self.dqn.target, feed_dict={
      self.dqn.next_state: next_state,
      self.dqn.reward: np.array([2.0]),
      self.dqn.done: np.array([False])
    })[0]
    self.assertAlmostEqual(target, 1.0 + 0.99 * np.max(next_q), delta=1e-4)

    target = self.sess.run(self.dqn.target, feed_dict={
      self.dqn.next_state: next_state,
      self.dqn.reward: np.array([1.0]),
      self.dqn.done: np.array([True])
    })[0]
    self.assertAlmostEqual(target, 1.0, delta=1e-4)

    target = self.sess.run(self.dqn.target, feed_dict={
      self.dqn.next_state: next_state,
      self.dqn.reward: np.array([0.0]),
      self.dqn.done: np.array([False])
    })[0]
    self.assertAlmostEqual(target, 0.99 * np.max(next_q), delta=1e-4)


def main():
  unittest.main()


if __name__ == '__main__':
  main()

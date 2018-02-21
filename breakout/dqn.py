import tensorflow as tf
import numpy as np


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
        self.copy_ops.append(tf.assign(target_vars[i], train_vars[i]))
        assert target_vars[i] not in tf.trainable_variables()
        assert train_vars[i] in tf.trainable_variables()

    with tf.name_scope('loss'):
      target = self.reward + config.gamma * \
        tf.cast(tf.logical_not(self.done), tf.float32) * \
        tf.reduce_max(self.next_q_values, axis=1)

      action_mask = tf.one_hot(self.action,
        config.action_size, name='action_mask')
      y = tf.reduce_sum(action_mask * self.q_values, axis=1, name='y')
      diff = y - target

      if use_huber:
        # Huber's loss
        error = tf.where(tf.abs(diff) < 1.0,
          0.5 * tf.square(diff), tf.abs(diff) - 0.5)
        self.loss = tf.reduce_mean(error, name='loss')
      else:
        self.loss = tf.reduce_mean(tf.square(diff), name='loss')

      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(config.learning_rate,
        trainable=False, name='learning_rate')
      optimizer = tf.train.RMSPropOptimizer(
        self.learning_rate,
        decay=config.decay,
        momentum=config.momentum,
        epsilon=config.eps)
      self.train_ops = optimizer.minimize(self.loss)

      tf.summary.scalar('learning_rate', self.learning_rate)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _setup_inputs(self):
    c = self.config
    self.state = tf.placeholder(dtype=tf.float32, name='state',
      shape=[None, c.input_height, c.input_width, c.skip])
    self.next_state = tf.placeholder(dtype=tf.float32, name='next_state',
      shape=[None, c.input_height, c.input_width, c.skip])
    self.reward = tf.placeholder(dtype=tf.float32, name='reward', shape=[None])
    self.done = tf.placeholder(dtype=tf.bool, name='done', shape=[None])
    self.action = tf.placeholder(dtype=tf.int32, name='action',
      shape=[None])

  def inference(self, inputs, trainable=True):
    with tf.name_scope('norm'):
      inputs = tf.div(inputs, 255.0)
      images = tf.split(inputs, [1, 1, 1, 1], axis=3)
      for i in range(4):
        tf.summary.image('input_images_%d' % i, images[i])

    initializer = tf.variance_scaling_initializer()
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=4, kernel_size=8,
        trainable=trainable, padding='VALID',
        weights_initializer=initializer)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(conv, 64, stride=2, kernel_size=4,
        trainable=trainable, padding='VALID',
        weights_initializer=initializer)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(conv, 64, stride=1, kernel_size=3,
        trainable=trainable, padding='VALID',
        weights_initializer=initializer)

    with tf.name_scope('fully_connected'):
      fc = tf.contrib.layers.fully_connected(
        tf.contrib.layers.flatten(conv), 512, trainable=trainable,
        weights_initializer=initializer)

    with tf.name_scope('output'):
      w = tf.get_variable('ow', shape=[512, self.config.action_size],
        trainable=trainable,
        initializer=initializer)
      b = tf.get_variable('ob', shape=[self.config.action_size],
        trainable=trainable,
        initializer=tf.zeros_initializer())
      outputs = tf.add(tf.matmul(fc, w), b, name='q_values')
    return outputs


def main():
  config = DQNConfig()
  dqn = DQN(config, False)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    state = np.random.uniform(0, 255, [1, 84, 84, 4])
    print(state.shape)
    q, nq = sess.run([dqn.q_values, dqn.next_q_values], feed_dict={
      dqn.state: state,
      dqn.next_state: state,
    })
    print(q)
    print(nq)
    sess.run(dqn.copy_ops)

    state = np.random.uniform(0, 255, [1, 84, 84, 4])
    q, nq = sess.run([dqn.q_values, dqn.next_q_values], feed_dict={
      dqn.state: state,
      dqn.next_state: state,
    })
    print(q)
    print(nq)


if __name__ == '__main__':
  main()

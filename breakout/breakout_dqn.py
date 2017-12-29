import tensorflow as tf
import numpy as np
import gym
import random
import coloredlogs
import logging
import threading
import signal
from collections import deque
from argparse import ArgumentParser


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('breakout')
logger.setLevel(logging.INFO)


class DQN(object):
  def __init__(self, learning_rate=1e-3):
    self._setup_inputs()

    with tf.variable_scope('dqn'):
      q_values = self.inference(self.state)
      self.q_values = q_values
    tf.summary.histogram('q_values', q_values)

    with tf.variable_scope('dqn', reuse=True):
      next_q_values = self.inference(self.next_state)
    tf.summary.histogram('next_q_values', next_q_values)

    with tf.name_scope('loss'):
      target = self.reward + tf.cast(tf.logical_not(self.done), tf.float32) * \
        tf.reduce_sum(next_q_values * self.action_mask, axis=1)
      y = tf.reduce_sum(q_values * self.action_mask)
      self.loss = tf.reduce_mean(tf.square(y - target))
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(learning_rate, trainable=False,
        name='learning_rate')
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

      tf.summary.scalar('learning_rate', self.learning_rate)

  def _setup_inputs(self):
    self.state = tf.placeholder(dtype=tf.float32, name='state',
      shape=[None, 210, 160, 3])
    self.next_state = tf.placeholder(dtype=tf.float32, name='next_state',
      shape=[None, 210, 160, 3])
    self.reward = tf.placeholder(dtype=tf.float32, name='reward', shape=[None])
    self.done = tf.placeholder(dtype=tf.bool, name='done', shape=[None])
    self.action_mask = tf.placeholder(dtype=tf.float32, name='action_mask',
      shape=[None, 4])

  def inference(self, inputs):
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 16, stride=1, kernel_size=3,
        weights_initializer=tf.random_normal_initializer(stddev=0.004))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 32, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 64, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(pool, 128, stride=1, kernel_size=3,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('fully_connected'):
      connect_shape = pool.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(pool, [-1, connect_size]), 64,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      outputs = tf.contrib.layers.fully_connected(fc, 4, activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
    return outputs

  def train(self, sess, states, actions, next_states, rewards, done):
    _, loss = sess.run([self.train_ops, self.loss], feed_dict={
      self.state: states,
      self.next_state: next_states,
      self.action_mask: actions,
      self.reward: rewards,
      self.done: done,
    })
    return loss

  def get_action(self, sess, state):
    q_values = sess.run(self.q_values, feed_dict={
      self.state: np.expand_dims(state, axis=0)
    })
    return q_values


def epsilon_greedy(q_values, epsilon):
  max_p = np.argmax(q_values)
  prob = np.ones(shape=[4]) * epsilon / 4.0
  prob[max_p] += 1.0 - epsilon
  return np.random.choice(np.arange(4), p=prob)


class Trainer(object):
  def __init__(self, args):
    self.dqn = DQN()
    self.replay_buffer = deque()
    self.max_epoches = args.max_epoches
    self.batch_size = args.batch_size
    self.display_epoches = args.display_epoches
    self.replay_buffer_size = args.replay_buffer_size

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    logger.info('initializing variables...')
    self.sess.run(tf.global_variables_initializer())
    self.running = True

  def __del__(self):
    self.running = False

  def train(self):
    logger.info('waiting for batch...')
    while len(self.replay_buffer) < self.batch_size:
      pass

    logger.info('start training...')
    for epoch in range(self.max_epoches + 1):
      batch = random.sample(self.replay_buffer, self.batch_size)
      state_batch = np.array([b[0] for b in batch])
      action_batch = np.array([[1 if i == b[1] else 0 for i in range(4)]
        for b in batch])
      next_state_batch = np.array([b[2] for b in batch])
      reward_batch = np.array([b[3] for b in batch])
      done_batch = np.array([b[4] for b in batch])

      loss = self.dqn.train(self.sess,
        state_batch, action_batch, next_state_batch,
        reward_batch, done_batch)
      if epoch % self.display_epoches == 0:
        q_values = self.sess.run(self.dqn.q_values, feed_dict={
          self.dqn.state: state_batch
        })
        logger.info('%d. loss: %f, max Q: %f',
          epoch, loss, np.max(q_values))

      if not self.running:
        break
    logger.info('training session stop')
    logger.info('closing session...')
    self.sess.close()

  def add_step(self, step):
    self.replay_buffer.append(step)

    if len(self.replay_buffer) > self.replay_buffer_size:
      self.replay_buffer.pop()

  def start(self):
    self.task = threading.Thread(target=self.train)
    self.task.start()

  def predict_action(self, state):
    if self.running:
      return self.dqn.get_action(self.sess, state)
    else:
      return 0


def run_episode(args, env):
  trainer = Trainer(args)
  trainer.start()

  def stop(signum, frame):
    logger.info('stopping...')
    trainer.running = False
  signal.signal(signal.SIGINT, stop)

  epsilon = 0.6
  for episode in range(args.max_episodes + 1):
    state = env.reset()
    step = 0
    while True:
      action_prob = trainer.predict_action(state)
      action = epsilon_greedy(action_prob, epsilon)
      next_state, reward, done, _ = env.step(action)
      #  if reward:
      #    reward = 100
      #  else:
      #    reward = 1

      trainer.add_step([state, action, next_state, reward, done])

      if args.render == 'True':
        env.render()
      state = next_state
      if done:
        logger.info('%d. episode, final step: %d, epsilon: %f',
          episode, step, epsilon)
        break

      if episode % 10 == 0 and episode != 0:
        epsilon *= 0.9

      step += 1

    if not trainer.running:
      break


def main():
  parser = ArgumentParser()
  parser.add_argument('--render', dest='render', default='True',
    help='render')
  parser.add_argument('--replay-buffer-size', dest='replay_buffer_size',
    type=int, default=20000, help='max replay buffer size')
  parser.add_argument('--learning-rate', dest='learning_rate', type=float,
    default=1e-3, help='learning rate for training')
  parser.add_argument('--batch-size', dest='batch_size', type=int,
    default=16, help='batch size for training')
  parser.add_argument('--max-episodes', dest='max_episodes', type=int,
    default=10000, help='max episode to run')
  parser.add_argument('--max-epoches', dest='max_epoches', type=int,
    default=200000, help='max epoches to train model')
  parser.add_argument('--display-epoches', dest='display_epoches', type=int,
    default=10, help='epoches to display training result')
  parser.add_argument('--save-epoches', dest='save_epoches', type=int,
    default=10000, help='epoches to save training result')
  parser.add_argument('--summary-epoches', dest='summary_epoches', type=int,
    default=10, help='epoches to save training summary')
  parser.add_argument('--decay-epoches', dest='decay_epoches', type=int,
    default=10000, help='epoches to decay learning rate for training')
  parser.add_argument('--keep-prob', dest='keep_prob', type=float,
    default=0.8, help='keep probability for dropout')
  parser.add_argument('--saving', dest='saving', type=str,
    default='False', help='rather to save the training result')
  args = parser.parse_args()

  env = gym.make('Breakout-v0')
  run_episode(args, env)


if __name__ == '__main__':
  main()

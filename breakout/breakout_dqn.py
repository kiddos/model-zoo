import tensorflow as tf
import numpy as np
import gym
import random
import coloredlogs
import logging
import threading
import signal
import os
import sys
import copy
from PIL import Image
from collections import deque
from argparse import ArgumentParser


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('breakout')
logger.setLevel(logging.INFO)


class DQN(object):
  def __init__(self, learning_rate=1e-4, discount_factor=0.99):
    self._setup_inputs()

    with tf.variable_scope('train'):
      q_values = self.inference(self.state)
      self.q_values = q_values
    tf.summary.histogram('q_values', q_values)

    with tf.variable_scope('target'):
      next_q_values = self.inference(self.next_state, False)
      self.next_q_values = next_q_values
    tf.summary.histogram('next_q_values', next_q_values)

    with tf.name_scope('copy_ops'):
      train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'train')
      target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target')
      self.copy_ops = []
      assert len(train_vars) == len(target_vars)
      for i in range(len(train_vars)):
        self.copy_ops.append(tf.assign(target_vars[i], train_vars[i]))

    with tf.name_scope('loss'):
      max_index = tf.argmax(next_q_values, axis=1)
      max_target = tf.one_hot(max_index, 4)
      target = discount_factor * \
        tf.expand_dims(tf.cast(tf.logical_not(self.done),
          tf.float32), axis=1) * \
        next_q_values * max_target + \
        tf.expand_dims(self.reward, axis=1) * max_target

      action_mask = tf.one_hot(self.action, 4)
      y = action_mask * q_values
      #  self.loss = tf.reduce_mean(tf.square(y - target))
      diff = tf.reduce_sum(y - target, axis=1)
      diff_abs = tf.abs(diff)
      condition = tf.cast(tf.less_equal(diff_abs, 1.0), tf.float32)
      error = tf.square(diff * condition) / 2.0 + \
        (diff_abs - 0.5) * (1.0 - condition)
      self.loss = tf.reduce_mean(error)
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(learning_rate, trainable=False,
        name='learning_rate')
      optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
        momentum=0.95, epsilon=0.01)
      self.train_ops = optimizer.minimize(self.loss)

      tf.summary.scalar('learning_rate', self.learning_rate)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _setup_inputs(self):
    self.state = tf.placeholder(dtype=tf.float32, name='state',
      shape=[None, 65, 68, 1])
    self.next_state = tf.placeholder(dtype=tf.float32, name='next_state',
      shape=[None, 65, 68, 1])
    self.reward = tf.placeholder(dtype=tf.float32, name='reward', shape=[None])
    self.done = tf.placeholder(dtype=tf.bool, name='done', shape=[None])
    self.action = tf.placeholder(dtype=tf.int32, name='action',
      shape=[None])

  def inference(self, inputs, trainable=True):
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 16, stride=2, kernel_size=8,
        trainable=trainable,
        weights_initializer=tf.random_normal_initializer(stddev=0.001))

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(conv, 32, stride=2, kernel_size=4,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('fully_connected'):
      connect_shape = conv.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(conv, [-1, connect_size]), 256, trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      outputs = tf.contrib.layers.fully_connected(fc, 4, activation_fn=None,
        trainable=trainable,
        weights_initializer=tf.random_uniform_initializer(-0.001, 0.001))
    return outputs

  def update_target(self, sess):
    sess.run(self.copy_ops)

  def train(self, sess, states, actions, next_states, rewards, done):
    _, loss = sess.run([self.train_ops, self.loss], feed_dict={
      self.state: states,
      self.next_state: next_states,
      self.action: actions,
      self.reward: rewards,
      self.done: done,
    })
    return loss

  def get_summary(self, sess, states, actions, next_states, rewards, done):
    return sess.run(self.summary, feed_dict={
      self.state: states,
      self.next_state: next_states,
      self.action: actions,
      self.reward: rewards,
      self.done: done,
    })

  def get_action(self, sess, state):
    q_values = sess.run(self.next_q_values, feed_dict={
      self.next_state: np.expand_dims(state, axis=0)
    })
    return q_values


def epsilon_greedy(q_values, epsilon):
  max_p = np.argmax(q_values)
  prob = np.ones(shape=[4]) * epsilon / 4.0
  prob[max_p] += 1.0 - epsilon
  return np.random.choice(np.arange(4), p=prob)


class Trainer(object):
  def __init__(self, args):
    self.replay_buffer = deque()

    self.max_epoches = args.max_epoches
    self.batch_size = args.batch_size
    self.display_epoches = args.display_epoches
    self.summary_epoches = args.summary_epoches
    self.save_epoches = args.save_epoches
    self.replay_buffer_size = args.replay_buffer_size
    self.saving = (args.saving == 'True')
    self.update_frequency = args.update_frequency
    self.init_replay_buffer_size = args.init_replay_buffer_size

    if self.saving:
      logger.info('saving model...')
      if not os.path.isdir('breakout-dqn'):
        os.mkdir('breakout-dqn')
      self.checkpoint = os.path.join('breakout-dqn', 'dqn')

    self.dqn = DQN(learning_rate=args.learning_rate,
      discount_factor=args.discount_factor)
    if self.saving:
      self.saver = tf.train.Saver()
      self.summary_writer = tf.summary.FileWriter(
        os.path.join('breakout-dqn', 'summary'), tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    logger.info('initializing variables...')
    self.sess.run(tf.global_variables_initializer())
    self.running = True
    self.start_training = False

  def __del__(self):
    self.running = False

  def get_batch(self, buffers):
    batch = random.sample(buffers, self.batch_size)
    state_batch = np.array([b[0] for b in batch])
    action_batch = np.array([b[1] for b in batch])
    next_state_batch = np.array([b[2] for b in batch])
    reward_batch = np.array([b[3] for b in batch])
    done_batch = np.array([b[4] for b in batch])
    return state_batch, action_batch, next_state_batch, reward_batch, done_batch

  def shuffle(self, state_batch, action_batch,
      next_state_batch, reward_batch, done_batch):
    index = np.random.permutation(np.arange(len(state_batch)))[:self.batch_size]
    return state_batch[index, :], \
      action_batch[index, :], \
      next_state_batch[index, :], \
      reward_batch[index], \
      done_batch[index]

  def train(self):
    self.update_target()

    logger.info('waiting for batch...')
    while len(self.replay_buffer) < self.init_replay_buffer_size:
      pass
    self.start_training = True

    logger.info('start training...')
    epoch = 0
    while self.running:
      state_batch, action_batch, next_state_batch, \
        reward_batch, done_batch = self.get_batch(self.replay_buffer)

      loss = self.dqn.train(self.sess,
        state_batch, action_batch, next_state_batch,
        reward_batch, done_batch)
      if epoch % self.display_epoches == 0:
        q_values = self.sess.run(self.dqn.q_values, feed_dict={
          self.dqn.state: state_batch
        })
        logger.info('%d. loss: %f, max Q: %f',
          epoch, loss, np.max(q_values))

      if epoch == 0:
        q_values, next_q_values = self.sess.run(
          [self.dqn.next_q_values, self.dqn.q_values], feed_dict={
            self.dqn.state: state_batch,
            self.dqn.next_state: next_state_batch,
          })
        logger.info('q values mean: %s, stddev: %s',
          str(np.mean(q_values, axis=0)), str(np.std(q_values, axis=0)))
        logger.info('next q values mean: %s, stddev: %s',
          str(np.mean(q_values, axis=0)), str(np.std(next_q_values, axis=0)))

      if epoch % self.save_epoches == 0 and epoch != 0 and self.saving:
        logger.info('saving model...')
        self.saver.save(self.sess, self.checkpoint, global_step=epoch)

      if epoch % self.summary_epoches == 0 and epoch != 0 and self.saving:
        summary = self.dqn.get_summary(self.sess,
          state_batch, action_batch, next_state_batch,
          reward_batch, done_batch)
        self.summary_writer.add_summary(summary)

      if epoch % self.update_frequency == 0 and epoch != 0:
        self.update_target()

      epoch += 1

    logger.info('training session stop')
    logger.info('closing session...')
    self.sess.close()

  def update_target(self):
    logger.info('update target network...')
    sys.stdout.flush()
    self.dqn.update_target(self.sess)

  def add_step(self, step):
    self.replay_buffer.append(step)

    if len(self.replay_buffer) > self.replay_buffer_size:
      self.replay_buffer.popleft()

  def start(self):
    self.task = threading.Thread(target=self.train)
    self.task.start()

  def predict_action(self, state):
    if self.running:
      return self.dqn.get_action(self.sess, state)
    else:
      return 0


def process_image(state):
  image = Image.fromarray(state).crop([8, 32, 144, 162])
  image = image.resize([68, 65]).convert('L')
  return np.expand_dims(np.array(image), axis=2)


def run_episode(args, env):
  trainer = Trainer(args)
  trainer.start()

  def stop(signum, frame):
    logger.info('stopping...')
    trainer.running = False
  signal.signal(signal.SIGINT, stop)

  epsilon = 1.0
  for episode in range(args.max_episodes + 1):
    state = env.reset()
    # start the game
    env.step(1)

    state = process_image(state)

    step = 0
    total_reward = 0

    max_q = np.max(trainer.predict_action(state))
    while True:
      action_prob = trainer.predict_action(state)
      ma = np.max(action_prob)
      if ma > max_q: max_q = ma

      if random.random() < epsilon:
        action = env.action_space.sample()
      else:
        #  action = epsilon_greedy(action_prob[0], epsilon)
        action = np.argmax(action_prob[0, :])

      next_state, reward, done, info = env.step(action)
      if info['ale.lives'] < 5: done = True
      next_state = process_image(next_state)
      total_reward += reward

      trainer.add_step([state, action, next_state, reward, done])

      if args.render == 'True':
        env.render()
      state = next_state
      if done:
        logger.info('%d. steps: %d, epsilon: %f, total: %f, max Q: %f',
          episode, step, epsilon, total_reward, max_q)
        sys.stdout.flush()
        break

      step += 1

    if not trainer.running:
      break

    if trainer.start_training and episode % args.decay_epsilon == 0 \
        and episode != 0:
      epsilon *= 0.9
      if epsilon <= args.min_epsilon:
        epsilon = args.min_epsilon


def main():
  parser = ArgumentParser()
  parser.add_argument('--render', dest='render', default='True',
    help='render')

  parser.add_argument('--init-replay-buffer-size',
    dest='init_replay_buffer_size', type=int, default=10000,
    help='init replay buffer size before training start')
  parser.add_argument('--replay-buffer-size', dest='replay_buffer_size',
    type=int, default=100000, help='max replay buffer size')
  parser.add_argument('--max-episodes', dest='max_episodes', type=int,
    default=2000000, help='max episode to run')
  parser.add_argument('--update-frequency', dest='update_frequency',
    type=int, default=200, help='update target vars per episode')
  parser.add_argument('--decay-epsilon', dest='decay_epsilon',
    type=int, default=100, help='decay epsilon for epsilon greedy policy')
  parser.add_argument('--min-epsilon', dest='min_epsilon', type=float,
    default=0.1, help='minimum epsilon to decay to')
  parser.add_argument('--discoun-factor', dest='discount_factor',
    type=float, default=0.99, help='discount factor')

  parser.add_argument('--learning-rate', dest='learning_rate', type=float,
    default=1e-4, help='learning rate for training')
  parser.add_argument('--batch-size', dest='batch_size', type=int,
    default=32, help='batch size for training')
  parser.add_argument('--max-epoches', dest='max_epoches', type=int,
    default=200000, help='max epoches to train model')
  parser.add_argument('--display-epoches', dest='display_epoches', type=int,
    default=50, help='epoches to display training result')
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

  env = gym.make('BreakoutDeterministic-v4')
  run_episode(args, env)


if __name__ == '__main__':
  main()

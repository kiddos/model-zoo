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
from collections import deque
from argparse import ArgumentParser


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('cartpole')
logger.setLevel(logging.INFO)


class DQN(object):
  def __init__(self, learning_rate=1e-3, discount_factor=0.9):
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
      target = self.reward + discount_factor * \
        tf.cast(tf.logical_not(self.done), tf.float32) * \
        tf.reduce_max(next_q_values, axis=1)
      y = tf.reduce_sum(q_values * self.action_mask, axis=1)
      self.loss = tf.reduce_mean(tf.square(y - target))
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(learning_rate, trainable=False,
        name='learning_rate')
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      gradients = optimizer.compute_gradients(self.loss)
      gradients = [(tf.clip_by_value(g, -1, 1), v) for g, v in gradients]
      self.train_ops = optimizer.apply_gradients(gradients)

      tf.summary.scalar('learning_rate', self.learning_rate)

  def _setup_inputs(self):
    self.state = tf.placeholder(dtype=tf.float32, name='state',
      shape=[None, 4])
    self.next_state = tf.placeholder(dtype=tf.float32, name='next_state',
      shape=[None, 4])

    self.reward = tf.placeholder(dtype=tf.float32, name='reward', shape=[None])
    self.done = tf.placeholder(dtype=tf.bool, name='done', shape=[None])
    self.action_mask = tf.placeholder(dtype=tf.float32, name='action_mask',
      shape=[None, 2])

  def inference(self, inputs, trainable=True):
    with tf.name_scope('hidden1'):
      fc = tf.contrib.layers.fully_connected(inputs, 64, trainable=trainable,
        activation_fn=tf.nn.tanh,
        weights_initializer=tf.random_normal_initializer(stddev=0.1))

    #  with tf.name_scope('hidden2'):
    #    fc = tf.contrib.layers.fully_connected(fc, 128, trainable=trainable,
    #      activation_fn=tf.nn.relu,
    #      weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      outputs = tf.contrib.layers.fully_connected(fc, 2, activation_fn=None,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())
    return outputs

  def update_target(self, sess):
    sess.run(self.copy_ops)

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
    q_values = sess.run(self.next_q_values, feed_dict={
      self.next_state: np.expand_dims(state, axis=0)
    })
    return q_values


def epsilon_greedy(q_values, epsilon):
  if random.random() < epsilon:
    return random.randint(0, 1)
  else:
    return np.argmax(q_values[0, :])
  #  max_p = np.argmax(q_values)
  #  prob = np.ones(shape=[2]) * epsilon / 2.0
  #  prob[max_p] += 1.0 - epsilon
  #  return np.random.choice(np.arange(2), p=prob)


class Trainer(object):
  def __init__(self, args):
    self.replay_buffer = deque()
    self.max_epoches = args.max_epoches
    self.batch_size = args.batch_size
    self.display_epoches = args.display_epoches
    self.save_epoches = args.save_epoches
    self.replay_buffer_size = args.replay_buffer_size
    self.update_frequency = args.update_frequency

    self.saving = (args.saving == 'True')
    if self.saving:
      logger.info('saving model...')
      if not os.path.isdir('dqn'):
        os.mkdir('dqn')
      self.checkpoint = os.path.join('dqn', 'dqn')

    self.dqn = DQN(args.learning_rate, args.discount_factor)
    self.saver = tf.train.Saver()

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
    epoch = 0
    while True:
      batch = random.sample(self.replay_buffer, self.batch_size)
      state_batch = np.array([b[0] for b in batch])
      action_batch = np.array([[1 if i == b[1] else 0 for i in range(2)]
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

      if epoch % self.save_epoches == 0 and epoch != 0 and self.saving:
        logger.info('saving model...')
        self.saver.save(self.sess, self.checkpoint, global_step=epoch)

      if epoch % self.update_frequency == 0 and epoch != 0:
        self.dqn.update_target(self.sess)

      if not self.running:
        break

      epoch += 1

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

  def update_target(self):
    self.dqn.update_target(self.sess)


def run_episode(args, env):
  trainer = Trainer(args)
  trainer.update_target()
  trainer.start()

  def stop(signum, frame):
    logger.info('stopping...')
    trainer.running = False
  signal.signal(signal.SIGINT, stop)

  epsilon = 0.9
  for episode in range(args.max_episodes + 1):
    state = env.reset()
    step = 0
    total_reward = 0
    while True:
      action_prob = trainer.predict_action(state)
      action = epsilon_greedy(action_prob, epsilon)

      next_state, reward, done, _ = env.step(action)
      total_reward += reward
      trainer.add_step([state, action, next_state, reward, done])

      if args.render == 'True':
        env.render()
      state = next_state
      if done:
        if episode % args.display_episode == 0:
          logger.info('%d. episode, step: %d, epsilon: %f, total: %f, buffer: %d',
            episode, step, epsilon, total_reward, len(trainer.replay_buffer))
        sys.stdout.flush()
        break

      step += 1

    if episode % args.decay_epsilon == 0 and episode != 0:
      epsilon *= 0.9
      if epsilon <= args.min_epsilon:
        epsilon = args.min_epsilon

    if not trainer.running:
      break

  trainer.running = False


def main():
  parser = ArgumentParser()
  parser.add_argument('--render', dest='render', default='True',
    help='render')
  parser.add_argument('--decay-epsilon', dest='decay_epsilon', default=50,
    type=int, help='decay epsilon')
  parser.add_argument('--display-episode', dest='display_episode', default=1,
    type=int, help='display episode')
  parser.add_argument('--replay-buffer-size', dest='replay_buffer_size',
    type=int, default=30000, help='max replay buffer size')
  parser.add_argument('--update-frequency', dest='update_frequency',
    default=50, type=int, help='update frequency')
  parser.add_argument('--min-epsilon', dest='min_epsilon', default=0.01,
    type=float, help='min epsilon')
  parser.add_argument('--discount-factor', dest='discount_factor', default=0.9,
    type=float, help='discount factor')

  parser.add_argument('--learning-rate', dest='learning_rate', type=float,
    default=1e-3, help='learning rate for training')
  parser.add_argument('--batch-size', dest='batch_size', type=int,
    default=512, help='batch size for training')
  parser.add_argument('--max-episodes', dest='max_episodes', type=int,
    default=100000, help='max episode to run')
  parser.add_argument('--max-epoches', dest='max_epoches', type=int,
    default=200000, help='max epoches to train model')
  parser.add_argument('--display-epoches', dest='display_epoches', type=int,
    default=500, help='epoches to display training result')
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

  env = gym.make('CartPole-v0')
  run_episode(args, env)


if __name__ == '__main__':
  main()

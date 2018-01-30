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
import matplotlib.pyplot as plt


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('cartpole')
logger.setLevel(logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('version', 0, 'environment version')
tf.app.flags.DEFINE_boolean('render', False, 'render environment')
tf.app.flags.DEFINE_boolean('saving', False, 'saving the model')
tf.app.flags.DEFINE_integer('display_episode', 1, 'display per episode')
tf.app.flags.DEFINE_integer('summary_episode', 100, 'save summary per episode')

# hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
  'learning rate to train model')
tf.app.flags.DEFINE_integer('max_episode', 30000, 'number of epoches')
tf.app.flags.DEFINE_integer('decay_to_episode', 500, 'epsilon decay to')
tf.app.flags.DEFINE_float('discount_factor', 0.9, 'discount factor gamma')
tf.app.flags.DEFINE_integer('update_frequency', 6, 'update frequency')
tf.app.flags.DEFINE_float('min_epsilon', 0.1, 'minimum epsilon')
tf.app.flags.DEFINE_integer('batch_size', 512, 'batch size')
tf.app.flags.DEFINE_integer('replay_buffer_size', 60000, 'replay buffer size')
tf.app.flags.DEFINE_integer('init_replay_buffer_size', 6000,
  'replay buffer starting size')


class DQN(object):
  def __init__(self):
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
      target = self.reward + FLAGS.discount_factor * \
        tf.cast(tf.logical_not(self.done), tf.float32) * \
        tf.reduce_max(next_q_values, axis=1)
      y = tf.reduce_sum(q_values * self.action_mask, axis=1, name='y')
      self.loss = tf.reduce_mean(tf.square(y - target), name='loss')
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(FLAGS.learning_rate,
        trainable=False, name='learning_rate')
      #  optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      optimizer = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1.0)
      #  gradients = optimizer.compute_gradients(self.loss)
      #  gradients = [(tf.clip_by_value(g, -1, 1), v) for g, v in gradients]
      #  self.train_ops = optimizer.apply_gradients(gradients)
      self.train_ops = optimizer.minimize(self.loss)

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
        weights_initializer=tf.random_normal_initializer(stddev=0.1))

    with tf.name_scope('hidden2'):
      fc = tf.contrib.layers.fully_connected(fc, 64, trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      outputs = tf.contrib.layers.fully_connected(fc, 2, activation_fn=None,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())
    return outputs


def epsilon_greedy(q_values, epsilon):
  max_p = np.argmax(q_values)
  prob = np.ones(shape=[2]) * epsilon / 2.0
  prob[max_p] += 1.0 - epsilon
  return np.random.choice(np.arange(2), p=prob)


class Trainer(object):
  def __init__(self):
    self.replay_buffer = deque()
    self.dqn = DQN()
    self.epoch = 0

  def get_batch(self):
    batch = random.sample(self.replay_buffer, FLAGS.batch_size)
    state = np.array([b[0] for b in batch])
    action = np.array([[1 if i == b[1] else 0 for i in range(2)]
      for b in batch])
    next_state = np.array([b[2] for b in batch])
    reward = np.array([b[3] for b in batch])
    done = np.array([b[4] for b in batch])
    return state, action, next_state, reward, done

  def train(self, sess):
    if len(self.replay_buffer) < FLAGS.batch_size: return

    self.epoch += 1
    state_b, action_b, next_state_b, reward_b, done_b = self.get_batch()
    sess.run(self.dqn.train_ops, feed_dict={
      self.dqn.state: state_b,
      self.dqn.action_mask: action_b,
      self.dqn.next_state: next_state_b,
      self.dqn.reward: reward_b,
      self.dqn.done: done_b,
    })

  def compute_loss(self, sess):
    if len(self.replay_buffer) < FLAGS.batch_size: return 0

    state_b, action_b, next_state_b, reward_b, done_b = self.get_batch()
    return sess.run(self.dqn.loss, feed_dict={
      self.dqn.state: state_b,
      self.dqn.action_mask: action_b,
      self.dqn.next_state: next_state_b,
      self.dqn.reward: reward_b,
      self.dqn.done: done_b,
    })

  def max_q(self, sess):
    if len(self.replay_buffer) < FLAGS.batch_size: return 0

    state_b, action_b, next_state_b, reward_b, done_b = self.get_batch()
    return sess.run(tf.reduce_max(self.dqn.next_q_values), feed_dict={
      self.dqn.next_state: state_b
    })

  def predict_action(self, sess, state):
    return sess.run(self.dqn.next_q_values, feed_dict={
      self.dqn.next_state: np.expand_dims(state, axis=0)
    })

  def update_target(self, sess):
    sess.run(self.dqn.copy_ops)

  def add_step(self, step):
    self.replay_buffer.append(step)

    if len(self.replay_buffer) >= FLAGS.replay_buffer_size:
      self.replay_buffer.popleft()

  def ready(self):
    return len(self.replay_buffer) >= FLAGS.init_replay_buffer_size


def decay_epsilon(episode, to):
  factor = to / -np.log(FLAGS.min_epsilon)
  return np.exp(-episode / factor)


def run_episode(env):
  trainer = Trainer()

  # fill data
  while not trainer.ready():
    state = env.reset()
    while True:
      action = env.action_space.sample()
      next_state, reward, done, info = env.step(action)
      trainer.add_step([state, action, next_state, reward, done])
      state = next_state
      if done: break


  # start training
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initializing variables...')
    sess.run(tf.global_variables_initializer())

    epsilon = 1.0
    for episode in range(FLAGS.max_episode + 1):
      if episode % FLAGS.update_frequency == 0:
        trainer.update_target(sess)

      state = env.reset()
      total_reward = 0

      epsilon = decay_epsilon(episode, FLAGS.decay_to_episode)
      if epsilon < FLAGS.min_epsilon: epsilon = FLAGS.min_epsilon
      while True:
        action_prob = trainer.predict_action(sess, state)
        action = epsilon_greedy(action_prob, epsilon)

        next_state, reward, done, info = env.step(action)
        trainer.add_step([state, action, next_state, reward, done])
        state = next_state

        total_reward += reward

        trainer.train(sess)
        if FLAGS.render: env.render()
        if done:
          loss = trainer.compute_loss(sess)
          max_q = trainer.max_q(sess)

          if episode % FLAGS.display_episode == 0:
            logger.info('%d. epsilon: %f, total: %f, loss: %f, max Q: %f',
              episode, epsilon, total_reward, loss, max_q)
          break


def main():
  env = gym.make('CartPole-v%d' % FLAGS.version)
  run_episode(env)


if __name__ == '__main__':
  main()

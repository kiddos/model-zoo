import tensorflow as tf
import numpy as np
import gym
import os
import coloredlogs
import logging
import signal
import sys
import threading
import random
import math
from collections import deque
from argparse import ArgumentParser


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('pendulum')
logger.setLevel(logging.INFO)


def actor_inference(inputs, trainable=True):
  with tf.name_scope('hidden1'):
    fc = tf.contrib.layers.fully_connected(inputs, 128,
      activation_fn=tf.nn.tanh, trainable=trainable,
      weights_initializer=tf.random_normal_initializer(stddev=1.0))

  with tf.name_scope('hidden2'):
    fc = tf.contrib.layers.fully_connected(fc, 128,
      activation_fn=tf.nn.tanh, trainable=trainable,
      weights_initializer=tf.variance_scaling_initializer())

  with tf.name_scope('output'):
    output = tf.contrib.layers.fully_connected(fc, 1,
      activation_fn=tf.nn.tanh, trainable=trainable,
      weights_initializer=tf.variance_scaling_initializer())
  return output


def critic_inference(state_inputs, action_inputs, trainable=True):
  with tf.name_scope('state_input'):
    s = tf.contrib.layers.fully_connected(state_inputs, 128,
      activation_fn=None, trainable=trainable,
      weights_initializer=tf.random_normal_initializer(stddev=0.1))

  with tf.name_scope('action_inputs'):
    a = tf.contrib.layers.fully_connected(action_inputs, 128,
      activation_fn=None, trainable=trainable,
      weights_initializer=tf.random_normal_initializer(stddev=0.1))

  with tf.name_scope('hidden1'):
    fc = tf.nn.tanh(a + s)

  with tf.name_scope('hidden2'):
    fc = tf.contrib.layers.fully_connected(fc, 128,
      activation_fn=tf.nn.tanh, trainable=trainable,
      weights_initializer=tf.variance_scaling_initializer())

  with tf.name_scope('output'):
    output = tf.contrib.layers.fully_connected(fc, 1,
      activation_fn=None, trainable=trainable,
      weights_initializer=tf.random_normal_initializer(stddev=1.0),
      biases_initializer=tf.constant_initializer(1e-3))
  return output


class DDPG(object):
  def __init__(self, args):
    #  with tf.variable_scope('train_actor'):
    #    self.train_action = self.actor.inference(self.state)

    #  with tf.name_scope('copy_actors'):
    #    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
    #      'train_actor')
    #    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
    #      'target_actor')
    #    assert len(train_vars) == len(target_vars)
    #    self.copy_ops = []
    #    for i in range(len(train_vars)):
    #      self.copy_ops.append(tf.assign(target_vars[i],
    #        train_vars[i] * args.tau + target_vars[i] * (1.0 - args.tau)))


    with tf.name_scope('critic_inputs'):
      self._setup_inputs()

    with tf.variable_scope('train_critic'):
      self.q_values = critic_inference(self.state, self.action)

    with tf.variable_scope('target_actor'):
      self.next_actions = actor_inference(self.next_state, False)

    with tf.variable_scope('target_critic'):
      self.next_q_values = critic_inference(self.next_state,
        self.next_actions, False)

    with tf.variable_scope('train_actor'):
      self.action_prediction = actor_inference(self.state)

    with tf.name_scope('updates'):
      self.update_critic_ops = self.create_update_ops('train_critic',
        'target_critic', args.tau)
      self.update_actor_ops = self.create_update_ops('train_actor',
        'target_actor', args.tau)

    with tf.name_scope('copy'):
      self.copy_critic_ops = self.create_update_ops('train_critic',
        'target_critic', 1.0)
      self.copy_actor_ops = self.create_update_ops('train_actor',
        'target_actor', 1.0)

    with tf.name_scope('loss'):
      target = self.reward + args.discount_factor * \
        tf.cast(tf.logical_not(self.done), tf.float32) * \
        tf.reduce_sum(self.next_q_values, axis=1)
      y = tf.reduce_sum(self.q_values, axis=1)
      self.loss = tf.reduce_mean(tf.square(y - target))

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(args.learning_rate,
        trainable=False, name='learning_rate')
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.train_critic = optimizer.minimize(self.loss)

      action_grad = tf.gradients(self.q_values, self.action)
      actor_train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        'train_actor')
      actor_grad = tf.gradients(self.action_prediction, actor_train_vars,
        -action_grad[0])
      self.train_actor = optimizer.apply_gradients(
        zip(actor_grad, actor_train_vars))

  def _setup_inputs(self):
    self.state = tf.placeholder(dtype=tf.float32, name='state',
      shape=[None, 3])
    self.action = tf.placeholder(dtype=tf.float32, name='action',
      shape=[None, 1])
    self.done = tf.placeholder(dtype=tf.bool, name='done',
      shape=[None])
    self.reward = tf.placeholder(dtype=tf.float32, name='reward',
      shape=[None])
    self.next_state = tf.placeholder(dtype=tf.float32, name='next_state',
      shape=[None, 3])

  def create_update_ops(self, train_scope, target_scope, tau):
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
      train_scope)
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
      target_scope)
    assert len(train_vars) != 0
    assert len(train_vars) == len(target_vars)
    copy_ops = []
    for i in range(len(train_vars)):
      copy_ops.append(tf.assign(target_vars[i],
        train_vars[i] * tau + target_vars[i] * (1.0 - tau)))
    return copy_ops

  def train(self, sess, states, actions, next_states, rewards, done):
    _, _, loss = sess.run([self.train_actor, self.train_critic, self.loss],
      feed_dict={
        self.state: states,
        self.action: actions,
        self.next_state: next_states,
        self.reward: rewards,
        self.done: done,
      })
    return loss

  def update_targets(self, sess):
    sess.run([self.update_actor_ops, self.update_critic_ops])

  def copy_targets(self, sess):
    sess.run([self.copy_actor_ops, self.copy_critic_ops])


class Trainer(object):
  def __init__(self, args):
    self.replay_buffer = deque()
    self.replay_buffer_size = args.replay_buffer_size
    self.batch_size = args.batch_size
    self.display_epoches = args.display_epoches
    self.running = True
    self.ddpg = DDPG(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    logger.info('initializing variables...')
    self.sess.run(tf.global_variables_initializer())

  def train(self):
    logger.info('waiting for replay buffer to fill...')
    while len(self.replay_buffer) < self.batch_size:
      pass

    self.ddpg.copy_targets(self.sess)

    logger.info('start training...')
    epoch = 0
    while True:
      batch = random.sample(self.replay_buffer, self.batch_size)
      state_batch = np.array([b[0] for b in batch])
      action_batch = np.array([b[1] for b in batch])
      next_state_batch = np.array([b[2] for b in batch])
      reward_batch = np.array([b[3] for b in batch])
      done_batch = np.array([b[4] for b in batch])

      if epoch == 0:
        q_values, action_prob = self.sess.run(
          [self.ddpg.next_q_values, self.ddpg.action_prediction], feed_dict={
            self.ddpg.state: state_batch,
            self.ddpg.next_state: next_state_batch,
          })
        logger.info('q values mean: %s, stddev: %s',
          str(np.mean(q_values, axis=0)), str(np.std(q_values, axis=0)))
        logger.info('action mean: %s, stddev: %s',
          str(np.mean(action_prob, axis=0)), str(np.std(action_prob, axis=0)))
        sys.stdout.flush()

      loss = self.ddpg.train(self.sess,
        state_batch, action_batch, next_state_batch,
        reward_batch, done_batch)

      self.ddpg.update_targets(self.sess)

      if epoch % self.display_epoches == 0:
        q_values = self.sess.run(self.ddpg.next_q_values, feed_dict={
          self.ddpg.next_state: next_state_batch,
        })
        logger.info('%d. loss: %f, max Q: %f', epoch, loss, np.max(q_values))
        sys.stdout.flush()

      if not self.running:
        break

      epoch += 1

    logger.info('closing session...')
    self.sess.close()

  def start(self):
    self.task = threading.Thread(target=self.train)
    self.task.start()

  def add_step(self, step):
    self.replay_buffer.append(step)

    if len(self.replay_buffer) > self.replay_buffer_size:
      self.replay_buffer.popleft()

  def get_action(self, state):
    return self.sess.run(self.ddpg.action_prediction, feed_dict={
      self.ddpg.state: np.expand_dims(state, axis=0)
    })[0]


def train(args):
  env = gym.make('Pendulum-v0')
  trainer = Trainer(args)
  trainer.start()
  render = (args.render == 'True')

  def stop(signum, frame):
    trainer.running = False
  signal.signal(signal.SIGINT, stop)

  random_chance = 0.8
  for episode in range(args.max_episode + 1):
    state = env.reset()
    total_reward = steps = 0
    while True:
      if random.random() < random_chance:
        action = env.action_space.sample()
      else:
        action = trainer.get_action(state)

      next_state, reward, done, _ = env.step(action)
      total_reward += reward
      trainer.add_step([state, action, next_state, reward, done])

      if render:
        env.render()

      if done:
        if episode % 10 == 0:
          logger.info('%d. total: %d, steps: %d, chance: %f',
            episode, total_reward, steps, random_chance)
          sys.stdout.flush()
        break
      state = next_state
      steps += 1

      if not trainer.running:
        break

    if episode % 10 == 0:
      random_chance *= 0.9


def main():
  parser = ArgumentParser()

  parser.add_argument('--discount-factor', dest='discount_factor',
    type=float, default=0.99, help='discount factor')
  parser.add_argument('--tau', dest='tau', type=float,
    default=1e-3, help='controls how much to update target network')
  parser.add_argument('--replay-buffer-size', dest='replay_buffer_size',
    type=int, default=10000, help='max replay buffer size')
  parser.add_argument('--max-episode', dest='max_episode',
    default=10000, type=int, help='max episode to run')
  parser.add_argument('--render', dest='render', default='False',
    help='render scene')

  parser.add_argument('--learning-rate', dest='learning_rate', type=float,
    default=1e-3, help='learning rate for training')
  parser.add_argument('--batch-size', dest='batch_size', type=int,
    default=32, help='batch size for training')
  parser.add_argument('--max-epoches', dest='max_epoches', type=int,
    default=100000, help='max epoches to train')
  parser.add_argument('--display-epoches', dest='display_epoches', type=int,
    default=100, help='epoches to display training result')
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

  train(args)


if __name__ == '__main__':
  main()

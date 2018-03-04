import tensorflow as tf
import numpy as np
import random
import coloredlogs
import logging
import threading
import signal
import os
import gym
from PIL import Image
from collections import deque
from argparse import ArgumentParser

from environment import get_training_env
from replay_buffer import ReplayBuffer
from dqn import DQN, DQNConfig


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('saving', False, 'saving model')
tf.app.flags.DEFINE_boolean('render', False, 'render environment')
tf.app.flags.DEFINE_string('environment', 'BreakoutDeterministic-v0',
  'openai gym environment to run')

# hyperparameters
tf.app.flags.DEFINE_integer('init_replay_buffer_size', 50000,
  'replay buffer starting size')
tf.app.flags.DEFINE_integer('replay_buffer_size', 300000,
  'replay buffer max size')
tf.app.flags.DEFINE_integer('max_episodes', 600000,
  'number of episodes to run')
tf.app.flags.DEFINE_integer('update_frequency', 10000,
  'update target network per episode')
tf.app.flags.DEFINE_integer('decay_to_epoch', 1000000,
  'decay epsilon until epoch')
tf.app.flags.DEFINE_float('min_epsilon', 0.1, 'min epsilon to decay to')
tf.app.flags.DEFINE_float('gamma', 0.99, 'discount factor')
tf.app.flags.DEFINE_float('learning_rate', 0.00025, 'learning rate to train')
tf.app.flags.DEFINE_float('decay', 0.95,
  'decay factor for next gradients for RMSProp')
tf.app.flags.DEFINE_float('momentum', 0.95, 'squred momentum for RMSProp')
tf.app.flags.DEFINE_float('eps', 0.01, 'eps for avoiding zero for RMSProp')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size to train')
tf.app.flags.DEFINE_integer('image_width', 84, 'input image width')
tf.app.flags.DEFINE_integer('image_height', 84, 'input image height')
tf.app.flags.DEFINE_bool('use_huber', True, 'use huber loss')
tf.app.flags.DEFINE_integer('skip', 4, 'skip frame')
tf.app.flags.DEFINE_integer('history_length', 4, 'history length')

tf.app.flags.DEFINE_integer('display_episode', 1, 'display result per episode')
tf.app.flags.DEFINE_integer('save_episode', 6000, 'save model per episode')
tf.app.flags.DEFINE_integer('summary_episode', 10, 'save summary per episode')


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('breakout')
logger.setLevel(logging.INFO)


class Trainer(object):
  def __init__(self):
    self.replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size,
        FLAGS.image_width, FLAGS.image_height, FLAGS.history_length)

    config = DQNConfig()
    config.learning_rate = FLAGS.learning_rate
    config.gamma = FLAGS.gamma
    config.decay = FLAGS.decay
    config.momentum = FLAGS.momentum
    config.eps = FLAGS.eps
    config.input_width = FLAGS.image_width
    config.input_height = FLAGS.image_height
    config.skip = FLAGS.skip
    self.dqn = DQN(config, FLAGS.use_huber)

  def train(self, sess):
    states, actions, next_states, rewards, done = \
        self.replay_buffer.sample(FLAGS.batch_size)
    sess.run(self.dqn.train_ops, feed_dict={
      self.dqn.state: states,
      self.dqn.next_state: next_states,
      self.dqn.action: actions,
      self.dqn.reward: rewards,
      self.dqn.done: done,
    })

  def update_target(self, sess):
    sess.run(self.dqn.copy_ops)

  def predict_action(self, sess, state):
    return sess.run(self.dqn.q_values, feed_dict={
      self.dqn.state: np.expand_dims(state, axis=0)
    })[0]

  def compute_loss(self, sess):
    states, actions, next_states, rewards, done = \
        self.replay_buffer.sample(FLAGS.batch_size)
    return sess.run(self.dqn.loss, feed_dict={
      self.dqn.state: states,
      self.dqn.next_state: next_states,
      self.dqn.action: actions,
      self.dqn.reward: rewards,
      self.dqn.done: done,
    })

  def ave_q_values(self, sess):
    states, _, _, _, _ = self.replay_buffer.sample(FLAGS.batch_size)
    max_q = np.max(sess.run(self.dqn.q_values, feed_dict={
      self.dqn.state: states,
    }), axis=1)
    return np.mean(max_q)

  def get_summary(self, sess):
    states, actions, next_states, rewards, done = \
        self.replay_buffer.sample(FLAGS.batch_size)
    return sess.run(self.dqn.summary, feed_dict={
      self.dqn.state: states,
      self.dqn.next_state: next_states,
      self.dqn.action: actions,
      self.dqn.reward: rewards,
      self.dqn.done: done,
    })

  def ready(self):
    return self.replay_buffer.current_size >= FLAGS.init_replay_buffer_size


def epsilon_greedy(trainer, sess, state, epsilon, env):
  if random.random() < epsilon:
    return env.action_space.sample()
  else:
    action_prob = trainer.predict_action(sess, state)
    return np.argmax(action_prob)


def decay_epsilon(epoch, to):
  #  factor = to / -np.log(FLAGS.min_epsilon)
  #  epsilon = np.exp(-epoch / factor)
  #  if epsilon < FLAGS.min_epsilon: epsilon = FLAGS.min_epsilon
  epsilon = 1.0 - (1.0 - FLAGS.min_epsilon) * epoch / to
  if epsilon < FLAGS.min_epsilon: epsilon = FLAGS.min_epsilon
  return epsilon


def prepare_folder():
  index = 0
  folder = os.path.join('/tmp', 'breakout_%d' % index)
  while os.path.isdir(folder):
    index += 1
    folder = os.path.join('/tmp', 'breakout_%d' % index)
  return folder


def run_episode(env):
  trainer = Trainer()

  # fill replay buffer
  logger.info('filling replay buffer...')
  state = env.reset()
  while not trainer.ready():
    while True:
      action = env.action_space.sample()
      next_state, reward, done, lives = env.step(action)
      trainer.replay_buffer.add(state, action, reward, done)
      state = next_state
      if done:
        env.reset()
        break
  logger.info('replay buffer size: %d', trainer.replay_buffer.current_size)

  if FLAGS.saving:
    folder = prepare_folder()
    saver = tf.train.Saver(max_to_keep=30)
    summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
      tf.get_default_graph())

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initializing variables')
    sess.run(tf.global_variables_initializer())
    trainer.update_target(sess)

    total_rewards = deque(maxlen=200)

    max_total_reward = 0
    epoch = 0
    actions = [0 for _ in range(env.action_space.n)]
    for episode in range(FLAGS.max_episodes + 1):
      epsilon = decay_epsilon(epoch, FLAGS.decay_to_epoch)

      step = 0
      total_reward = 0
      while True:
        if epoch % FLAGS.update_frequency == 0:
          trainer.update_target(sess)

        recent_states = trainer.replay_buffer.recent_state(state)

        action = epsilon_greedy(trainer, sess, recent_states, epsilon, env)
        actions[action] += 1
        next_state, reward, done, lives = env.step(action)

        trainer.replay_buffer.add(state, action, reward, done)
        state = next_state

        step += 1
        total_reward += reward

        trainer.train(sess)
        epoch += 1

        if FLAGS.render == 'True':
          env.render()

        if done:
          env.reset()

          total_rewards.append(total_reward)

          if total_reward > max_total_reward * 0.8 and FLAGS.saving:
            saver.save(sess, os.path.join(folder, 'breakout'),
              global_step=episode)

          if total_reward > max_total_reward:
            max_total_reward = total_reward

          if episode % FLAGS.display_episode == 0:
            loss = trainer.compute_loss(sess)
            ave_q = trainer.ave_q_values(sess)

            logger.info('%d. steps: %d, eps: %f, total: %f, max R: %f',
              episode, step, epsilon, total_reward, max_total_reward)
            logger.info('%d. average Q: %f, loss: %f', epoch, ave_q, loss)
            logger.info('ave: %f, max: %f',
                sum(total_rewards) / len(total_rewards), max(total_rewards))
            logger.info('actions: %s', str(actions))
            actions = [0 for _ in range(env.action_size)]

          if FLAGS.saving and \
              episode % FLAGS.summary_episode == 0:
            summary = trainer.get_summary(sess)
            summary_writer.add_summary(summary, global_step=episode)
          break

      if FLAGS.saving and episode % FLAGS.save_episode == 0:
        saver.save(sess, os.path.join(folder, 'breakout'),
          global_step=episode)

    if FLAGS.saving:
      saver.save(sess, os.path.join(folder, 'breakout'),
        global_step=episode)


def main(_):
  env = get_training_env(FLAGS.environment, FLAGS.image_width,
    FLAGS.image_height)
  run_episode(env)


if __name__ == '__main__':
  tf.app.run()

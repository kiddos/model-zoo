import tensorflow as tf
import numpy as np
import gym
import random
import coloredlogs
import logging
import threading
import signal
import os
from PIL import Image
from collections import deque
from argparse import ArgumentParser


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('saving', False, 'saving model')
tf.app.flags.DEFINE_boolean('render', False, 'render environment')
tf.app.flags.DEFINE_string('environment', 'Breakout-v0',
  'openai gym environment to run')

# hyperparameters
tf.app.flags.DEFINE_integer('init_replay_buffer_size', 10000,
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
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size to train')
tf.app.flags.DEFINE_integer('image_width', 84, 'input image width')
tf.app.flags.DEFINE_integer('image_height', 84, 'input image height')
tf.app.flags.DEFINE_bool('use_huber', True, 'use huber loss')

tf.app.flags.DEFINE_integer('display_episode', 1, 'display result per episode')
tf.app.flags.DEFINE_integer('save_episode', 1000, 'save model per episode')
tf.app.flags.DEFINE_integer('summary_episode', 10, 'save summary per episode')


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('breakout')
logger.setLevel(logging.INFO)


class DQN(object):
  def __init__(self):
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

    with tf.name_scope('loss'):
      target = self.reward + FLAGS.gamma * \
        tf.cast(tf.logical_not(self.done), tf.float32) * \
        tf.reduce_max(self.next_q_values, axis=1)

      action_mask = tf.one_hot(self.action, 4, name='action_mask')
      y = tf.reduce_sum(action_mask * self.q_values, axis=1, name='y')

      if FLAGS.use_huber:
        # Huber's loss
        diff = y - target
        diff_abs = tf.abs(diff)
        condition = tf.cast(tf.less_equal(diff_abs, 1.0), tf.float32)
        error = tf.square(diff * condition) / 2.0 + \
          (diff_abs - 0.5) * (1.0 - condition)
        self.loss = tf.reduce_mean(error, name='loss')
      else:
        self.loss = tf.reduce_mean(tf.square(y - target), name='loss')

      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False,
        name='learning_rate')
      optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
        FLAGS.decay, FLAGS.momentum, FLAGS.eps)
      self.train_ops = optimizer.minimize(self.loss)
      #  grads = optimizer.compute_gradients(self.loss)
      #  grads = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in grads]
      #  self.train_ops = optimizer.apply_gradients(grads)

      tf.summary.scalar('learning_rate', self.learning_rate)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _setup_inputs(self):
    self.state = tf.placeholder(dtype=tf.float32, name='state',
      shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
    self.next_state = tf.placeholder(dtype=tf.float32, name='next_state',
      shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
    self.reward = tf.placeholder(dtype=tf.float32, name='reward', shape=[None])
    self.done = tf.placeholder(dtype=tf.bool, name='done', shape=[None])
    self.action = tf.placeholder(dtype=tf.int32, name='action',
      shape=[None])

  def inference(self, inputs, trainable=True):
    with tf.name_scope('norm'):
      inputs = tf.div(inputs, 255.0)

    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 16, stride=4, kernel_size=8,
        trainable=trainable, padding='VALID',
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(conv, 32, stride=2, kernel_size=4,
        trainable=trainable, padding='VALID',
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
        weights_initializer=tf.glorot_uniform_initializer())
    return outputs


class Trainer(object):
  def __init__(self):
    self.replay_buffer = deque()
    self.dqn = DQN()

  def get_batch(self, batch_size):
    batch = random.sample(self.replay_buffer, batch_size)
    state_batch = np.array([b[0] for b in batch])
    action_batch = np.array([b[1] for b in batch])
    next_state_batch = np.array([b[2] for b in batch])
    reward_batch = np.array([b[3] for b in batch])
    done_batch = np.array([b[4] for b in batch])
    return state_batch, action_batch, next_state_batch, reward_batch, done_batch

  def train(self, sess):
    state_batch, action_batch, next_state_batch, \
      reward_batch, done_batch = self.get_batch(FLAGS.batch_size)
    sess.run(self.dqn.train_ops, feed_dict={
      self.dqn.state: state_batch,
      self.dqn.action: action_batch,
      self.dqn.next_state: next_state_batch,
      self.dqn.reward: reward_batch,
      self.dqn.done: done_batch,
    })

  def update_target(self, sess):
    sess.run(self.dqn.copy_ops)

  def add_step(self, step):
    self.replay_buffer.append(step)

    if len(self.replay_buffer) > FLAGS.replay_buffer_size:
      self.replay_buffer.popleft()

  def predict_action(self, sess, state):
    return sess.run(self.dqn.next_q_values, feed_dict={
      self.dqn.next_state: np.expand_dims(state, axis=0)
    })[0]

  def compute_loss(self, sess):
    state_batch, action_batch, next_state_batch, \
      reward_batch, done_batch = self.get_batch(FLAGS.batch_size)
    return sess.run(self.dqn.loss, feed_dict={
      self.dqn.state: state_batch,
      self.dqn.next_state: next_state_batch,
      self.dqn.action: action_batch,
      self.dqn.reward: reward_batch,
      self.dqn.done: done_batch,
    })

  def max_q_values(self, sess):
    state_batch, action_batch, next_state_batch, \
      reward_batch, done_batch = self.get_batch(FLAGS.batch_size)
    return np.max(sess.run(self.dqn.next_q_values, feed_dict={
      self.dqn.next_state: state_batch,
    }))

  def get_summary(self, sess):
    state_batch, action_batch, next_state_batch, \
      reward_batch, done_batch = self.get_batch(FLAGS.batch_size)
    return sess.run(self.dqn.summary, feed_dict={
      self.dqn.state: state_batch,
      self.dqn.next_state: next_state_batch,
      self.dqn.action: action_batch,
      self.dqn.reward: reward_batch,
      self.dqn.done: done_batch,
    })

  def ready(self):
    return len(self.replay_buffer) >= FLAGS.init_replay_buffer_size


def epsilon_greedy(trainer, sess, state, epsilon):
  if random.random() < epsilon:
    return random.randint(0, 3)
  else:
    action_prob = trainer.predict_action(sess, state)
    return np.argmax(action_prob)


def process_image(state):
  image = Image.fromarray(state).crop([8, 32, 152, 210])
  #  image = image.resize([68, 65]).convert('L')
  image = image.resize([FLAGS.image_width, FLAGS.image_height],
    Image.NEAREST).convert('L')
  img = np.expand_dims(np.array(image, dtype=np.uint8), axis=2)
  return img


def decay_epsilon(epoch, to):
  factor = to / -np.log(FLAGS.min_epsilon)
  return np.exp(-epoch / factor)


def run_episode(env):
  trainer = Trainer()

  # fill replay buffer
  logger.info('filling replay buffer...')
  while not trainer.ready():
    state = env.reset()
    state = process_image(state)
    while True:
      action = env.action_space.sample()
      next_state, reward, done, info = env.step(action)
      next_state = process_image(next_state)
      trainer.add_step([state, action, next_state, reward, done])
      state = next_state
      if done: break
  logger.info('replay buffer size: %d', len(trainer.replay_buffer))

  if FLAGS.saving:
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('/tmp/breakout/summary',
      tf.get_default_graph())

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initializing variables')
    sess.run(tf.global_variables_initializer())
    trainer.update_target(sess)

    max_total_reward = 0
    epoch = 0
    for episode in range(FLAGS.max_episodes + 1):
      state = env.reset()
      #  start the game
      env.step(1)
      state = process_image(state)

      epsilon = decay_epsilon(epoch, FLAGS.decay_to_epoch)
      if epsilon < FLAGS.min_epsilon: epsilon = FLAGS.min_epsilon

      step = 0
      total_reward = 0
      while True:
        if epoch % FLAGS.update_frequency == 0:
          trainer.update_target(sess)

        action = epsilon_greedy(trainer, sess, state, epsilon)
        next_state, reward, done, info = env.step(action)
        if info['ale.lives'] < 5: done = True
        next_state = process_image(next_state)
        trainer.add_step([state, action, next_state, reward, done])

        step += 1
        total_reward += reward
        state = next_state

        trainer.train(sess)
        epoch += 1

        if FLAGS.render == 'True':
          env.render()
        if done:
          if total_reward > max_total_reward: max_total_reward = total_reward
          if episode % FLAGS.display_episode == 0:
            loss = trainer.compute_loss(sess)
            max_qs = trainer.max_q_values(sess)

            logger.info('%d. steps: %d, eps: %f, total: %f, max R: %f',
              episode, step, epsilon, total_reward, max_total_reward)
            logger.info('%d. max Q: %f, loss: %f',
              epoch, max_qs, loss)

          if FLAGS.saving and \
              episode % FLAGS.save_episode == 0 and episode != 0:
            saver.save(sess, '/tmp/breakout/breakout', global_step=episode)

          if FLAGS.saving and \
              episode % FLAGS.summary_episode == 0:
            summary = trainer.get_summary(sess)
            summary_writer.add_summary(summary, global_step=episode)
          break


def main(_):
  env = gym.make(FLAGS.environment)
  run_episode(env)


if __name__ == '__main__':
  tf.app.run()

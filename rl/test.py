import logging
import os
from collections import deque

import numpy as np
import tensorflow as tf
import gym

from replay_buffer import ReplayBuffer
from dqn import DQN
from data_collector import Collector


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('saving', False, 'saving model')
tf.app.flags.DEFINE_integer('display_epoch', 100, 'epoches to display result')
tf.app.flags.DEFINE_integer('save_epoch', 1000, 'epoches to save model')
tf.app.flags.DEFINE_integer('summary_epoch', 10000, 'epoches to save summary')
tf.app.flags.DEFINE_string('environment', 'CartPole-v0', 'environment')
tf.app.flags.DEFINE_integer('num_actions', 2, 'number of actions')

# hyperparameter
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size to train')
tf.app.flags.DEFINE_integer('epsilon_stop_epoch', 20000, 'max epoches to train')
tf.app.flags.DEFINE_integer('max_epoch', 100000, 'max epoches to train')
tf.app.flags.DEFINE_integer('copy_epoch', 100, 'max epoches to train')
tf.app.flags.DEFINE_float('min_epsilon', 0.05, 'min epsilon for epsilon greedy')
tf.app.flags.DEFINE_integer('history_size', 1, 'input size of image')
tf.app.flags.DEFINE_integer('replay_buffer_size', 50000, 'batch size to train')


logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)


MODEL_PATH = 'model'


def run_episode(env, replay_buffer, policy):
  state = env.reset()
  total_reward = 0
  while True:
    action = policy(np.expand_dims(state, axis=0))
    next_state, reward, done, info = env.step(action)

    if replay_buffer is not None:
      replay_buffer.add(next_state, action, reward, done)
    state = next_state
    total_reward += reward
    if done:
      break
  return total_reward


def linear_model(inputs, name):
  with tf.variable_scope(name):
    hidden_size = 32
    with tf.name_scope('hidden_units'):
      fc = tf.contrib.layers.fully_connected(inputs, hidden_size,
        activation_fn=tf.nn.tanh,
        weights_initializer=tf.contrib.layers.variance_scaling_initializer())

    with tf.name_scope('output'):
      w = tf.get_variable(name + 'ow', shape=[hidden_size, FLAGS.num_actions],
        initializer=tf.contrib.layers.variance_scaling_initializer())
      b = tf.get_variable(name + 'ob', shape=[FLAGS.num_actions],
        initializer=tf.zeros_initializer())
      outputs = tf.add(tf.matmul(fc, w), b, name='q_values')
  return outputs


def main():
  input_shape = [4]
  replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size, input_shape, 1, np.float32)
  replay_buffer.HISTORY_SIZE = 1
  env = gym.make(FLAGS.environment)
  dqn = DQN(input_shape, linear_model, FLAGS.num_actions)

  collector = Collector(gym.make(FLAGS.environment), replay_buffer)
  logger.info('pre-loading replay buffer...')
  collector.prepare()
  logger.info('done.')

  logger.info('start collecting asyn...')
  collector.start()

  if FLAGS.saving:
    model_path = os.path.join(MODEL_PATH, FLAGS.environment)
    checkpoint = os.path.join(model_path, FLAGS.environment)
    summary_path = os.path.join(model_path, 'summary')

    logger.info('preparing saver...')
    saver = tf.train.Saver()
    logger.info('preparing summary writer...')
    summary_writer = tf.summary.FileWriter(summary_path, tf.get_default_graph())

  with tf.Session() as sess:
    logger.info('initializing variables...')
    sess.run(tf.global_variables_initializer())

    def greedy(state):
      p = sess.run(dqn.best_actions, feed_dict={
        dqn.state: state
      })
      return p[0]

    for epoch in range(FLAGS.max_epoch):
      if epoch % FLAGS.copy_epoch == 0:
        sess.run(dqn.copy)

      epsilon = max(1.0 - (1.0 - FLAGS.min_epsilon) /
        FLAGS.epsilon_stop_epoch * epoch, FLAGS.min_epsilon)

      def epsilon_greedy(state):
        chance = np.random.uniform()
        if chance < epsilon:
          return env.action_space.sample()
        return greedy(state)

      collector.set_policy(epsilon_greedy)

      states, next_states, actions, rewards, dones = \
        replay_buffer.next(FLAGS.batch_size)

      _, loss = sess.run([dqn.train_ops, dqn.loss], feed_dict={
        dqn.state: states,
        dqn.next_state: next_states,
        dqn.action: actions,
        dqn.reward: rewards,
        dqn.done: dones,
      })

      if epoch % FLAGS.display_epoch == 0:
        total_reward = run_episode(env, None, greedy)
        logger.info('%d. loss: %f, epsilon: %f, ave reward: %f, greedy: %f',
          epoch, loss, epsilon, collector.get_average_rewards(), total_reward)

      if FLAGS.saving and epoch > 0 and epoch % FLAGS.save_epoch == 0:
        logger.info('saving session...')
        saver.save(sess, checkpoint, global_step=epoch)

      if FLAGS.saving and epoch % FLAGS.summary_epoch == 0:
        summary = sess.run(dqn.summary, feed_dict={
          dqn.state: states,
          dqn.next_state: next_states,
          dqn.action: actions,
          dqn.reward: rewards,
          dqn.done: dones,
        })
        summary_writer.add_summary(summary, global_step=epoch)


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    logger.info('stop training.')

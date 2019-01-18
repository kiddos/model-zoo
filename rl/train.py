import logging
import os
from collections import deque

import numpy as np
import tensorflow as tf
import gym

from gym_wrapper import wrap_env
from replay_buffer import ReplayBuffer
from dqn import DQN
from dqn import atari_model
from data_collector import Collector


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('saving', False, 'saving model')
tf.app.flags.DEFINE_integer('display_epoch', 100, 'epoches to display result')
tf.app.flags.DEFINE_integer('save_epoch', 1000, 'epoches to save model')
tf.app.flags.DEFINE_integer('summary_epoch', 10000, 'epoches to save summary')
tf.app.flags.DEFINE_string('environment', 'Breakout-v0', 'environment to train')
tf.app.flags.DEFINE_integer('num_actions', 4, 'number of actions')

# hyperparameter
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size to train')
tf.app.flags.DEFINE_integer('epsilon_stop_epoch', 100000, 'max epoches to train')
tf.app.flags.DEFINE_integer('max_epoch', 300000, 'max epoches to train')
tf.app.flags.DEFINE_integer('copy_epoch', 100, 'max epoches to train')
tf.app.flags.DEFINE_float('min_epsilon', 0.1, 'min epsilon for epsilon greedy')
tf.app.flags.DEFINE_integer('input_size', 84, 'input size of image')
tf.app.flags.DEFINE_integer('history_size', 4, 'history size')
tf.app.flags.DEFINE_integer('replay_buffer_size', 1000000, 'replay buffer size')


logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)


MODEL_PATH = 'model'


def run_episode(env, replay_buffer, policy):
  state_stack = deque(maxlen=FLAGS.history_size)
  for _ in range(FLAGS.history_size):
    state_stack.append(
      np.zeros(shape=[FLAGS.input_size, FLAGS.input_size], dtype=np.uint8))

  state = env.reset()
  state_stack.append(state)
  total_reward = 0
  while True:
    input_state = np.expand_dims(np.stack(state_stack, axis=2), axis=0)
    action = policy(input_state)
    next_state, reward, done, info = env.step(action)

    if replay_buffer is not None:
      replay_buffer.add(state, action, reward, done)
    state = next_state
    state_stack.append(state)
    total_reward += reward
    if done:
      break
  return total_reward


def main():
  input_shape = [FLAGS.input_size, FLAGS.input_size]
  replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size, input_shape,
    FLAGS.history_size)

  env = wrap_env(gym.make(FLAGS.environment))
  input_shape += [FLAGS.history_size]
  dqn = DQN(input_shape, atari_model, FLAGS.num_actions)

  collector = Collector(wrap_env(gym.make(FLAGS.environment)), replay_buffer)
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
      return sess.run(dqn.best_actions, feed_dict={
        dqn.state: state
      })[0]

    for epoch in range(FLAGS.max_epoch + 1):
      if epoch % FLAGS.copy_epoch == 0:
        sess.run(dqn.copy)

      epsilon = max(1.0 - (1.0 - FLAGS.min_epsilon) /
        FLAGS.epsilon_stop_epoch * epoch, FLAGS.min_epsilon)

      def epsilon_greedy(state):
        chance = np.random.uniform()
        if chance < epsilon:
          return env.action_space.sample()
        return greedy(state)

      if epoch == 0:
        logger.info('filling replay buffer...')
        while not replay_buffer.ready():
          run_episode(env, replay_buffer, epsilon_greedy)
        logger.info('done.')

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
          epoch, loss, epsilon, collector.get_average_rewards(),
          total_reward)

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

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


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('saving', False, 'saving model')
tf.app.flags.DEFINE_integer('display_epoch', 100, 'epoches to display result')
tf.app.flags.DEFINE_integer('save_epoch', 1000, 'epoches to save model')
tf.app.flags.DEFINE_integer('summary_epoch', 10000, 'epoches to save summary')
tf.app.flags.DEFINE_string('environment', 'Breakout-v0', 'epoches to save summary')

# hyperparameter
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size to train')
tf.app.flags.DEFINE_integer('epsilon_stop_epoch', 100000, 'max epoches to train')
tf.app.flags.DEFINE_integer('max_epoch', 300000, 'max epoches to train')
tf.app.flags.DEFINE_integer('copy_epoch', 100, 'max epoches to train')
tf.app.flags.DEFINE_float('min_epsilon', 0.1, 'min epsilon for epsilon greedy')
tf.app.flags.DEFINE_integer('input_size', 84, 'input size of image')
tf.app.flags.DEFINE_integer('history_size', 4, 'input size of image')
tf.app.flags.DEFINE_integer('replay_buffer_size', 1000000, 'batch size to train')


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
  replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size,
    FLAGS.input_size, FLAGS.input_size)

  env = wrap_env(gym.make(FLAGS.environment))
  input_shape = [FLAGS.input_size, FLAGS.input_size, FLAGS.history_size]
  dqn = DQN(input_shape, atari_model)

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

    def predict(state):
      return sess.run(dqn.best_action, feed_dict={
        dqn.state: state
      })[0]

    def greedy(state):
      return predict(state)

    total_rewards = deque(maxlen=100)
    for epoch in range(FLAGS.max_epoch):
      if epoch % FLAGS.copy_epoch == 0:
        sess.run(dqn.copy)

      epsilon = max(1.0 - (1.0 - FLAGS.min_epsilon) /
        FLAGS.epsilon_stop_epoch * epoch, FLAGS.min_epsilon)

      def epsilon_greedy(state):
        chance = np.random.uniform()
        if chance < epsilon:
          return env.action_space.sample()
        return predict(state)

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

      total_reward = run_episode(env, replay_buffer, epsilon_greedy)
      total_rewards.append(total_reward)

      if epoch % FLAGS.display_epoch == 0:
        total_reward = run_episode(env, None, greedy)
        logger.info('%d. loss: %f, epsilon: %f, ave reward: %f, greedy: %f',
          epoch, loss, epsilon, float(sum(total_rewards)) / len(total_rewards),
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

import tensorflow as tf
import numpy as np
import gym
import logging
import random
from argparse import ArgumentParser
from collections import deque
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger('cartpole ddpg')
logger.setLevel(logging.INFO)


DISCOUNT_FACTOR = 0.99


class Critic(object):
  HIDDEN_SIZE = 128
  TAU = 1e-3
  LEARNING_RATE = 1e-3

  def __init__(self):
    self.learning_rate = tf.Variable(self.LEARNING_RATE, trainable=False)
    self._setup_inputs()

    with tf.variable_scope('critic_train'):
      self.train_output = self._inference(True)
    with tf.variable_scope('critic_target'):
      self.target_output = self._inference(False)

    self._setup_copy_ops()

    with tf.name_scope('action_gradients'):
      self.action_grad = tf.gradients(self.train_output, self.action)

    with tf.name_scope('loss'):
      not_done = tf.cast(tf.logical_not(self.done), tf.float32)
      target = self.reward + not_done * self.next_q_value
      out = tf.reduce_sum(self.action * self.train_output, axis=1)
      self.loss = tf.reduce_mean(tf.square(out - target))

    with tf.name_scope('optimization'):
      learning_rate = tf.Variable(self.LEARNING_RATE, trainable=False)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

  def _setup_inputs(self):
    with tf.name_scope('critic_inputs'):
      self.state = tf.placeholder(dtype=tf.float32,
        shape=[None, 4], name='state')
      self.action = tf.placeholder(dtype=tf.float32,
        shape=[None, 2], name='action')
      self.reward = tf.placeholder(dtype=tf.float32,
        shape=[None], name='reward')
      self.done = tf.placeholder(dtype=tf.bool,
        shape=[None], name='done')
      self.next_q_value = tf.placeholder(dtype=tf.float32,
        shape=[None], name='next_q_value')

  def _inference(self, trainable):
    with tf.name_scope('state_input'):
      s = tf.contrib.layers.fully_connected(self.state, self.HIDDEN_SIZE,
        trainable=trainable)

    with tf.name_scope('action_input'):
      a = tf.contrib.layers.fully_connected(self.action, self.HIDDEN_SIZE,
        trainable=trainable)

    with tf.name_scope('hidden'):
      h = tf.concat([s, a], axis=1)

    with tf.name_scope('output'):
      output = tf.contrib.layers.fully_connected(h, 2, activation_fn=None,
        trainable=trainable)
    return output

  def _setup_copy_ops(self):
    train_vars = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 'critic_train')
    target_vars = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 'critic_target')
    assert len(train_vars) == len(target_vars)

    with tf.name_scope('copy_vars'):
      self.copy_ops = []
      for i in range(len(target_vars)):
        self.copy_ops.append(tf.assign(target_vars[i], train_vars[i]))

    with tf.name_scope('update_vars'):
      self.update_ops = []
      for i in range(len(target_vars)):
        self.update_ops.append(tf.assign(target_vars[i],
          target_vars[i] * (1.0 - self.TAU) + train_vars[i] * self.TAU))

  def run_copy_ops(self, sess):
    sess.run(self.copy_ops)

  def run_update_ops(self, sess):
    sess.run(self.update_ops)

  def predict_batch(self, sess, state, action):
    return sess.run(self.target_output, feed_dict={
      self.state: state,
      self.action: action
    })

  def predict(self, sess, state, action):
    return sess.run(self.target_output, feed_dict={
      self.state: state,
      self.action: action
    })

  def get_action_grad(self, sess, state, action):
    return sess.run(self.action_grad, feed_dict={
      self.state: state,
      self.action: action
    })

  def train(self, sess, state, action, reward, done, next_q):
    _, loss = sess.run([self.train_ops, self.loss], feed_dict={
      self.state: state,
      self.action: action,
      self.reward: reward,
      self.done: done,
      self.next_q_value: next_q
    })
    return loss


class Actor(object):
  HIDDEN_SIZE = 128
  TAU = 1e-3
  LEARNING_RATE = 1e-3

  def __init__(self):
    self._setup_inputs()

    with tf.variable_scope('actor_train'):
      self.train_output = self._inference(True)
    with tf.variable_scope('actor_target'):
      self.target_output = self._inference(False)

    self._setup_copy_ops()

    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
      'actor_train')
    with tf.name_scope('optimization'):
      learning_rate = tf.Variable(self.LEARNING_RATE, trainable=False)
      grad = tf.gradients(self.train_output, train_vars, self.action_grad)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      self.train_ops = optimizer.apply_gradients(zip(grad, train_vars))
      self.decay_lr = tf.assign(learning_rate, learning_rate * 0.9)

  def _setup_inputs(self):
    with tf.name_scope('actor_inputs'):
      self.state = tf.placeholder(dtype=tf.float32, shape=[None, 4],
        name='state')
      self.action_grad = tf.placeholder(dtype=tf.float32, shape=[None, 2],
        name='acttion_gradients')

  def _setup_copy_ops(self):
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
      'actor_train')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
      'actor_target')

    with tf.name_scope('copy_ops'):
      self.copy_ops = []
      for i in range(len(train_vars)):
        self.copy_ops.append(tf.assign(target_vars[i], train_vars[i]))

    with tf.name_scope('update_ops'):
      self.update_ops = []
      for i in range(len(train_vars)):
        self.update_ops.append(tf.assign(target_vars[i],
          target_vars[i] * (1 - self.TAU) + train_vars[i] * self.TAU))

  def _inference(self, trainable):
    with tf.name_scope('fc1'):
      h1 = tf.contrib.layers.fully_connected(self.state, self.HIDDEN_SIZE,
        trainable=trainable)

    with tf.name_scope('fc2'):
      h2 = tf.contrib.layers.fully_connected(h1, self.HIDDEN_SIZE,
        trainable=trainable)

    with tf.name_scope('output'):
      output = tf.nn.softmax(
        tf.contrib.layers.fully_connected(h2, 2, activation_fn=None,
        trainable=trainable))
    return output

  def run_copy_ops(self, sess):
    sess.run(self.copy_ops)

  def run_update_ops(self, sess):
    sess.run(self.update_ops)

  def predict_batch(self, sess, state):
    return sess.run(self.target_output, feed_dict={
      self.state: state
    })

  def predict(self, sess, state):
    return sess.run(self.target_output, feed_dict={
      self.state: np.expand_dims(state, axis=0)
    })

  def train(self, sess, state, action_grad):
    sess.run(self.train_ops, feed_dict={
      self.state: state,
      self.action_grad: action_grad
    })


def epsilon_greedy_policy(actions_prob, epsilon):
  prob = np.ones(shape=[2], dtype=np.float32) * epsilon / 2
  #  print(prob)
  greedy_action = np.argmax(actions_prob)
  prob[greedy_action] += (1.0 - epsilon)
  action = np.random.choice(np.arange(2), p=prob)
  return action


def main():
  parser = ArgumentParser()
  parser.add_argument('--max-episodes', dest='max_episode',
    default=10000, type=int, help='max episode to run')
  parser.add_argument('--max-steps', dest='max_step',
    default=500, type=int, help='max step to run a episode')
  parser.add_argument('--batch-size', dest='batch_size',
    default=32, type=int, help='batch size to train policy gradient')
  parser.add_argument('--buffer-size', dest='buffer_size',
    default=10000, type=int, help='replay buffer max size')

  args = parser.parse_args()

  env_name = 'CartPole-v0'
  env = gym.make(env_name)

  critic = Critic()
  actor = Actor()

  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    critic.run_copy_ops(sess)
    actor.run_copy_ops(sess)

    replay_buffer = deque()

    train_epoch = 0
    epsilon = 0.9
    for episode in range(args.max_episode):
      state = env.reset()

      for t in range(args.max_step):
        action_pred = actor.predict(sess, state)
        action = epsilon_greedy_policy(action_pred[0], epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if len(replay_buffer) > args.batch_size:
          batch = random.sample(replay_buffer, args.batch_size)
          state_batch = np.array([b[0] for b in batch])
          action_batch = np.array([[1 if l == b[1] else 0 for l in range(2)]
            for b in batch])
          reward_batch = np.array([b[2] for b in batch])
          next_state_batch = np.array([b[3] for b in batch])
          done_batch = np.array([b[4] for b in batch])

          # train critic
          next_q_values = critic.predict_batch(sess, next_state_batch,
            action_batch)
          loss = critic.train(sess, state_batch, action_batch, reward_batch,
            done_batch, next_q_values[action_batch == 1])

          if train_epoch % 100 == 0:
            logger.info('%d. episode: %d | loss: %f, max q: %f' % (
              train_epoch, episode, loss, next_q_values.max()))

          # train actor
          action_grad = critic.get_action_grad(sess, state_batch, action_batch)
          actor.train(sess, state_batch, action_grad[0])

          critic.run_update_ops(sess)
          actor.run_update_ops(sess)

          train_epoch += 1

          while len(replay_buffer) > args.buffer_size:
            replay_buffer.popleft()

        if done:
          break
      # decay epsilon greedy policy
      epsilon *= 0.9


if __name__ == '__main__':
  main()

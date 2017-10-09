import tensorflow as tf
import numpy as np
import gym
import logging
import random
from argparse import ArgumentParser
from collections import deque
import matplotlib.pyplot as plt


logging.basicConfig()
logger = logging.getLogger('cartpole v0')
logger.setLevel(logging.INFO)


class Critic(object):
  def __init__(self, hidden_size, learning_rate=1e-3, gamma=0.99, tau=1e-3):
    self.hidden_size = hidden_size

    self.learning_rate = tf.Variable(learning_rate, trainable=False)
    self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * 0.9)
    self._setup_inputs()
    with tf.variable_scope('critic_train'):
      self.train_output = self._build_model(self.state, True)
    with tf.variable_scope('critic_target'):
      self.target_output = self._build_model(self.state, False)

    train_vars = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 'critic_train')
    target_vars = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 'critic_target')
    assert len(train_vars) == len(target_vars)

    with tf.name_scope('update'):
      self.update_target_vars = []
      for i in range(len(train_vars)):
        self.update_target_vars.append(tf.assign(
          target_vars[i], train_vars[i] * tau + target_vars[i] * (1.0 - tau)))
    with tf.name_scope('copy'):
      self.copy_train_weights = []
      for i in range(len(train_vars)):
        self.copy_train_weights.append(tf.assign(
          target_vars[i], train_vars[i]))

    with tf.name_scope('critic_loss'):
      y = (self.reward + tf.cast(tf.logical_not(self.done), tf.float32) *
        0.99 * self.q_value) * self.action
      self.loss = tf.reduce_mean(tf.square(y -
        self.action * self.train_output))
    with tf.name_scope('critic_optimize'):
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.train_op = optimizer.minimize(self.loss)
    with tf.name_scope('action_grad'):
      self.action_grad = tf.gradients(self.train_output, self.action)

  def _setup_inputs(self):
    with tf.name_scope('critic_inputs'):
      self.state = tf.placeholder(dtype=tf.float32,
        shape=[None, 4], name='critic_state')
      self.action = tf.placeholder(dtype=tf.float32,
        shape=[None, 2], name='critic_action')
      self.reward = tf.placeholder(dtype=tf.float32,
        shape=[None, 1], name='critic_reward')
      self.done = tf.placeholder(dtype=tf.bool,
        shape=[None, 1], name='critic_done')
      self.q_value = tf.placeholder(dtype=tf.float32,
        shape=[None, 2], name='critic_q_value')

  def _build_model(self, state, trainable):
    with tf.name_scope('hidden'):
      state_w = tf.get_variable(name='state_w', shape=[4, self.hidden_size],
        initializer=tf.random_normal_initializer(stddev=np.sqrt(0.5)),
        trainable=trainable)
      state_b = tf.get_variable(name='state_b', shape=[self.hidden_size],
        initializer=tf.constant_initializer(value=1e-3),
        trainable=trainable)
      action_w = tf.get_variable(name='action_w', shape=[2, self.hidden_size],
        initializer=tf.random_normal_initializer(stddev=1.0),
        trainable=trainable)
      action_b = tf.get_variable(name='action_b', shape=[self.hidden_size],
        initializer=tf.constant_initializer(value=1e-3),
        trainable=trainable)
      h = tf.concat([
        tf.nn.relu(tf.matmul(state, state_w) + state_b),
        tf.nn.relu(tf.matmul(self.action, action_w) + action_b)], axis=1)

    with tf.name_scope('output'):
      w = tf.get_variable(name='ow', shape=[self.hidden_size * 2, 2],
        initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 /
          self.hidden_size)),
        trainable=trainable)
      b = tf.get_variable(name='ob', shape=[2],
        initializer=tf.constant_initializer(value=1e-3),
        trainable=trainable)
      output = tf.matmul(h, w) + b
    return output

  def train(self, sess, state, action, reward, done, q_value):
    loss, _, q_values = sess.run(
      [self.loss, self.train_op, self.target_output], feed_dict={
        self.state: state,
        self.action: action,
        self.reward: reward,
        self.done: done,
        self.q_value: q_value
      })
    return loss, q_values

  def predict(self, sess, state, action):
    return sess.run(self.target_output, feed_dict={
      self.state: state,
      self.action: action
    })

  def update_target(self, sess):
    sess.run(self.update_target_vars)

  def copy_weights(self, sess):
    sess.run(self.copy_train_weights)

  def get_action_grad(self, sess, state, action, reward):
    return sess.run(self.action_grad, feed_dict={
      self.state: state,
      self.action: action,
      self.reward: reward
    })


class Actor(object):
  def __init__(self, hidden_size, learning_rate=1e-3, tau=1e-3):
    self.hidden_size = hidden_size
    self.learning_rate = tf.Variable(learning_rate, trainable=False)

    self._setup_inputs()
    with tf.variable_scope('actor_train'):
      self.train_output = self._build_model(True)
    with tf.variable_scope('actor_target'):
      self.target_output = self._build_model(False)

    with tf.name_scope('actor_loss'):
      train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        'actor_train')
      actor_grad = tf.gradients(self.train_output, train_vars,
        -self.action_grad)
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.apply_gradients(zip(actor_grad, train_vars))

    train_vars = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 'actor_train')
    target_vars = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 'actor_target')
    assert len(train_vars) == len(target_vars)

    with tf.name_scope('update'):
      self.update_target_vars = []
      for i in range(len(train_vars)):
        self.update_target_vars.append(tf.assign(
          target_vars[i], train_vars[i] * tau + target_vars[i] * (1.0 - tau)))
    with tf.name_scope('copy'):
      self.copy_train_weights = []
      for i in range(len(train_vars)):
        self.copy_train_weights.append(tf.assign(
          target_vars[i], train_vars[i]))


  def _setup_inputs(self):
    with tf.name_scope('actor_inputs'):
      self.state = tf.placeholder(dtype=tf.float32,
        shape=[None, 4,], name='actor_state')
      self.action_grad = tf.placeholder(dtype=tf.float32,
        shape=[None, 2], name='action_gradient')

  def _build_model(self, trainable):
    with tf.name_scope('hidden'):
      state_w = tf.get_variable(name='state_w',
        shape=[4, self.hidden_size],
        initializer=tf.random_normal_initializer(stddev=np.sqrt(1.0)),
        trainable=trainable)
      state_b = tf.get_variable(name='state_b',
        shape=[self.hidden_size],
        initializer=tf.constant_initializer(value=1.0),
        trainable=trainable)
      h = tf.nn.sigmoid(tf.matmul(self.state, state_w) + state_b)

    with tf.name_scope('action_output'):
      action_w = tf.get_variable(name='action_w', shape=[self.hidden_size, 2],
        initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 /
          self.hidden_size)),
        trainable=trainable)
      action_b = tf.get_variable(name='action_n', shape=[2],
        initializer=tf.constant_initializer(value=1.0),
        trainable=trainable)
      action = tf.nn.softmax(tf.matmul(h, action_w) + action_b)
    return action

  def update_target(self, sess):
    sess.run(self.update_target_vars)

  def copy_weights(self, sess):
    sess.run(self.copy_train_weights)

  def predict(self, sess, state):
    return sess.run(self.target_output, feed_dict={
      self.state: np.expand_dims(state, axis=0)
    })

  def predict_batch(self, sess, state):
    return sess.run(self.target_output, feed_dict={
      self.state: state
    })

  def train(self, sess, state, action_grad):
    _, q_value = sess.run([self.train_ops, self.target_output], feed_dict={
      self.state: state,
      self.action_grad: action_grad
    })
    return q_value


def epsilon_greedy_policy(actions_prob, epsilon):
  prob = np.ones(shape=[2], dtype=np.float32) * epsilon / 2
  #  print(prob)
  greedy_action = np.argmax(actions_prob)
  prob[greedy_action] += (1.0 - epsilon)
  action = np.random.choice(np.arange(2), p=prob)
  return action


def train(env, critic_hidden_size, actor_hidden_size,
    critic_learning_rate=1e-3,
    actor_learning_rate=1e-3,
    tau=1e-3,
    max_epoch=10000,
    max_step=500,
    batch_size=64,
    display_epoch=100,
    replay_buffer_size=1000,
    render=False):
  critic = Critic(critic_hidden_size)
  actor = Actor(actor_hidden_size)

  epsilon = 0.9
  train_epoch = 0
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # update target network
    critic.update_target(sess)
    actor.update_target(sess)

    #  figure = plt.figure()
    replay_buffer = deque()
    for epoch in range(max_epoch):
      state = env.reset()

      if epoch % 30 == 0 and epoch != 0:
        total_reward = 0
        for step in range(max_step):
          action_prob = actor.predict(sess, state)
          action = np.argmax(action_prob)
          next_state, reward, done, _ = env.step(action)
          total_reward += reward
          if render:
            env.render()
          state = next_state
          if done:
            logger.info('total reward: %d' % (total_reward))
            break
      else:
        q = []
        for step in range(max_step):
          action_prob = actor.predict(sess, state)
          action = epsilon_greedy_policy(action_prob, epsilon)
          next_state, reward, done, _ = env.step(action)
          #  print(state, action, reward, next_state, done)
          replay_buffer.append((state, action, reward, next_state, done))
          state = next_state
          q.append(critic.predict(sess, np.expand_dims(state, axis=0),
            np.array([[1 if a == action else 0 for a in range(2)]])))

          if len(replay_buffer) >= replay_buffer_size:
            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            done_batch = []
            for b in range(batch_size):
              batch = random.choice(replay_buffer)
              state_batch.append(batch[0])
              action_batch.append([1 if a == batch[1] else 0
                for a in range(2)])
              reward_batch.append([batch[2]])
              next_state_batch.append(batch[3])
              done_batch.append([batch[4]])
            state_batch = np.array(state_batch)
            action_batch = np.array(action_batch)
            reward_batch = np.array(reward_batch)
            next_state_batch = np.array(next_state_batch)
            done_batch = np.array(done_batch, dtype=np.bool)

            q_value_batch = critic.predict(sess, next_state_batch,
              np.squeeze(actor.predict_batch(sess, next_state_batch)))

            loss, q_values = critic.train(sess, state_batch,
              action_batch, reward_batch, done_batch, q_value_batch)
            action_grad = critic.get_action_grad(sess, state_batch,
              action_batch, reward_batch)
            actor.train(sess, state_batch,
              np.squeeze(action_grad, axis=0))
            if train_epoch % display_epoch == 0:
              logger.info('%d. episode: %d | critic loss: %f | q value: %s' %
                (train_epoch, epoch, loss, str(np.amax(q_values))))
              #  logger.info('action prob: %s' % (str(action_prob)))
            train_epoch += 1

            # update target network
            critic.update_target(sess)
            actor.update_target(sess)

          if done:
            break

          while len(replay_buffer) > replay_buffer_size * 2:
            replay_buffer.pop()

        #  # debugging
        #  if train_epoch % 100 == 0:
        #    q = np.squeeze(np.array(q))
        #    left, = plt.plot(np.arange(len(q)), q[:, 0],
        #      label='line 1 %d' % (train_epoch))
        #    right, = plt.plot(np.arange(len(q)), q[:, 1],
        #      label='line 2 %d' % (train_epoch))
        #    plt.legend(handles=[left, right])
        #    #  plt.show()
        #    plt.show(block=False)
        #    figure.canvas.draw()


def main():
  parser = ArgumentParser()
  parser.add_argument('--tau', dest='tau', default=1e-5,
    type=float, help='variable for updating target network')
  parser.add_argument('--critic-hidden-size', dest='critic_hidden_size',
    default=128, type=int, help='critic hidden size')
  parser.add_argument('--actor-hidden-size', dest='actor_hidden_size',
    default=128, type=int, help='actor hidden size')
  parser.add_argument('--critic-learning-rate', dest='critic_learning_rate',
    default=1e-3, type=float, help='critic learning rate')
  parser.add_argument('--actor-learning-rate', dest='actor_learning_rate',
    default=1e-4, type=float, help='actor learning rate')
  parser.add_argument('--max-epoch', dest='max_epoch',
    default=100000, type=int, help='max epoch to train')
  parser.add_argument('--max-step', dest='max_step',
    default=500, type=int, help='max step to run for each episode')
  parser.add_argument('--batch-size', dest='batch_size',
    default=64, type=int, help='batch size to train model')
  parser.add_argument('--render', dest='render',
    default=False, type=bool, help='display animation')
  args = parser.parse_args()

  env_name = 'CartPole-v1'
  env = gym.make(env_name)
  logger.info('%s: action space: %s' % (env_name, str(env.action_space)))
  logger.info('%s: observation space: %s' %
    (env_name, str(env.observation_space)))

  train(env, args.critic_hidden_size, args.actor_hidden_size,
    critic_learning_rate=args.critic_learning_rate,
    actor_learning_rate=args.actor_learning_rate,
    tau=args.tau,
    max_epoch=args.max_epoch,
    max_step=args.max_step,
    batch_size=args.batch_size,
    render=args.render)


if __name__ == '__main__':
  main()

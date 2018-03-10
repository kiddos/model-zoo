import tensorflow as tf
import numpy as np
import threading
import random
import unittest
import time
from collections import deque

from run import load_graph
from replay_buffer import ReplayBuffer
from environment import get_training_env


class ExperienceReplay(object):
  def __init__(self, env_name,
      replay_buffer_size, w, h, history_size, policy, decay_steps):
    self.epsilon = 1.0
    self.decay_steps = decay_steps
    self.policy = policy
    self.steps = 0

    self.replay_buffer = ReplayBuffer(replay_buffer_size, w, h, history_size)
    self.env = get_training_env(env_name, w, h)
    self.ave_rewards = deque(maxlen=100)
    self.max_reward = 0

  def run(self):
    self.running = True
    while self.running:
      state = self.env.reset()
      total_reward = 0
      while self.running:
        action = self.epsilon_greedy(state)
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.add(state, action, reward, done)
        state = next_state
        total_reward += reward

        if done:
          self.ave_rewards.append(total_reward)
          if total_reward >= self.max_reward:
            self.max_reward = total_reward
          break

  def init_replay_buffer(self, size):
    while self.replay_buffer.current_size < size:
      state = self.env.reset()
      while True:
        action = self.epsilon_greedy(state)
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.add(state, action, reward, done)
        state = next_state
        if done: break

  def set_epsilon(self, epsilon):
    self.epsilon = epsilon

  def start(self):
    self.task = threading.Thread(target=self.run)
    self.task.start()

  def stop(self):
    self.running = False
    if hasattr(self, 'task'):
      self.task.join()

  @property
  def average_reward(self):
    return sum(self.ave_rewards) / len(self.ave_rewards)

  @property
  def max_ave_reward(self):
    return max(self.ave_rewards)

  def epsilon_greedy(self, state):
    if random.random() < self.epsilon:
      return self.env.action_space.sample()
    else:
      return self.policy(self.replay_buffer.recent_state(state))


def get_graph(path):
  with tf.gfile.GFile(path, 'rb') as gf:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(gf.read())

    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def)
      return graph


class TestExperienceReplay(unittest.TestCase):
  def test_run(self):
    graph = get_graph('./models/breakout_1.pb')
    input_state = graph.get_tensor_by_name('import/state:0')
    q_values = graph.get_tensor_by_name('import/train/output/q_values:0')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=graph) as sess:
      def policy(state):
        action_prob = sess.run(q_values, feed_dict={
          input_state: np.expand_dims(state, axis=0)
        })
        action = np.argmax(action_prob[0, :])
        return action

      experience_replay = ExperienceReplay('BreakoutDeterministic-v0',
        100000, 84, 84, 4, policy, 1000)
      experience_replay.set_epsilon(0.01)

      experience_replay.start()
      time.sleep(10)
      print('average reward: %f' % experience_replay.average_reward)
      experience_replay.stop()


if __name__ == '__main__':
  unittest.main()

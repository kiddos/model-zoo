from collections import deque
from threading import Thread
import random
import time
import sys

import gym
import numpy as np

from replay_buffer import ReplayBuffer


class Collector(object):
  MIN_BUFFER_SIZE = 10

  def __init__(self, env, replay_buffer):
    self.env = env
    self.replay_buffer = replay_buffer
    self.running = True
    self.total_rewards = deque(maxlen=100)

    def random_policy(state):
      return self.env.action_space.sample()

    self.policy = random_policy

  def set_policy(self, policy):
    self.policy = policy

  def run_episode(self):
    state = self.env.reset()
    total_reward = 0
    while True:
      action = self.policy(np.expand_dims(state, axis=0))
      next_state, reward, done, info = self.env.step(action)
      self.replay_buffer.add(next_state, action, reward, done)
      state = next_state
      total_reward += reward
      if done:
        break
    return total_reward

  def run(self):
    while self.running:
      tr = self.run_episode()
      self.total_rewards.append(tr)

  def start(self):
    self.running = True
    self.task = Thread(target=self.run)
    self.task.daemon = True
    self.task.start()

  def stop(self):
    if hasattr(self, 'task'):
      self.running = False
      self.task.join(1)

  def ready(self):
    return self.replay_buffer.current_size >= self.MIN_BUFFER_SIZE

  def next(self, batch_size):
    return self.replay_buffer.next(batch_size)

  def prepare(self):
    while not self.ready():
      self.run_episode()

  def get_average_rewards(self):
    if len(self.total_rewards) == 0:
      return 0
    return float(sum(self.total_rewards)) / len(self.total_rewards)


def test():
  input_shape = [4]
  replay_buffer = ReplayBuffer(50000, input_shape)
  replay_buffer.HISTORY_SIZE = 1
  env = gym.make('CartPole-v0')

  collector = Collector(env, replay_buffer)
  print('loading replay buffer...')
  collector.prepare()
  print('done.')
  collector.start()

  try:
    while True:
      data = collector.next(32)
      print(data[0].shape)
  except Exception:
    print('stop')
  finally:
    collector.stop()


if __name__ == '__main__':
  test()

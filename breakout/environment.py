import gym
import numpy as np
from random import randint
from PIL import Image
from collections import deque
import time
import unittest


class EpisodicLifeEnv(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    self.lives = 0
    self.was_real_done = True

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    lives = info['ale.lives']
    if lives < self.lives:
      done = True
    self.lives = lives
    return state, reward, done, info

  def reset(self):
    if self.lives == 0:
      state = self.env.reset()
    else:
      state, _, _, info = self.env.step(0)
      self.lives = info['ale.lives']
    return state


class FireResetEnv(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    assert len(env.unwrapped.get_action_meanings()) >= 3

  def reset(self):
    self.env.reset()
    state, _, done, _ = self.env.step(1)
    if done: self.env.reset()
    state, _, done, _ = self.env.step(2)
    if done: self.env.reset()
    return state

  def step(self, action):
    return self.env.step(action)


def process_image(state, input_width, input_height):
  image = Image.fromarray(state).crop([8, 32, 152, 210])
  image = image.resize([input_width, input_height], Image.NEAREST).convert('L')
  img = np.array(image, dtype=np.uint8)
  return img


class MapState(gym.ObservationWrapper):
  def __init__(self, env, w, h):
    gym.ObservationWrapper.__init__(self, env)
    self.w, self.h = w, h
    self.observation_space = gym.spaces.Box(
      low=0, high=255, shape=(self.h, self.w), dtype=np.uint8)

  def observation(self, obs):
    return process_image(obs, self.w, self.h)


def get_training_env(name, w, h):
  env = gym.make(name)
  env.seed((int(time.time())))
  env = EpisodicLifeEnv(env)
  env = FireResetEnv(env)
  env = MapState(env, w, h)
  return env


class FireLiveLoss(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    self.lives = 0

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    lives = info['ale.lives']
    if lives < self.lives:
      state, _, _, info = self.env.step(0)
      state, _, _, info = self.env.step(1)
    self.lives = lives
    return state, reward, done, info


class HistoryStatesEnv(gym.Wrapper):
  def __init__(self, env, history_size):
    gym.Wrapper.__init__(self, env)
    self.history = deque(maxlen=history_size)

  def reset(self):
    state = self.env.reset()
    empty = np.zeros(self.observation_space.shape, np.uint8)
    for _ in range(self.history.maxlen - 1):
      self.history.append(empty)
    self.history.append(state)
    return self._history()

  def _history(self):
    return np.stack(self.history, axis=2)

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    self.history.append(state)
    return self._history(), reward, done, info


def get_test_env(name, w, h, history_size):
  env = gym.make(name)
  env.seed((int(time.time())))
  env = FireLiveLoss(env)
  env = MapState(env, w, h)
  env = HistoryStatesEnv(env, history_size)
  return env


class TestEnvironment(unittest.TestCase):
  def setUp(self):
    self.run_count = 100
    self.render = True

  #  def test_training_env(self):
  #    env = get_training_env('BreakoutDeterministic-v0', 84, 84)

  #    for i in range(self.run_count):
  #      state = env.reset()

  #      self.assertEqual(state.shape[0], 84)
  #      self.assertEqual(state.shape[1], 84)
  #      steps = 0

  #      total_reward = 0
  #      while True:
  #        action = randint(0, 3)
  #        next_state, reward, done, lives = env.step(action)

  #        self.assertEqual(next_state.shape[0], 84)
  #        self.assertEqual(next_state.shape[1], 84)

  #        steps += 1
  #        total_reward += reward

  #        if self.render:
  #          env.render()

  #        if done:
  #          break

  def test_test_env(self):
    env = get_test_env('BreakoutDeterministic-v0', 84, 84, 4)
    for i in range(self.run_count):
      state = env.reset()

      self.assertEqual(state.shape[0], 84)
      self.assertEqual(state.shape[1], 84)
      self.assertEqual(state.shape[2], 4)
      steps = 0

      total_reward = 0
      while True:
        action = randint(0, 3)
        next_state, reward, done, lives = env.step(action)

        self.assertEqual(next_state.shape[0], 84)
        self.assertEqual(next_state.shape[1], 84)
        self.assertEqual(next_state.shape[2], 4)

        steps += 1
        total_reward += reward
        state = next_state

        if self.render:
          env.render()

        if done:
          break


def main():
  unittest.main()


if __name__ == '__main__':
  main()

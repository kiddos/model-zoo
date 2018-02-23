import gym
import numpy as np
from random import randint
from PIL import Image
from collections import deque
import time
import cv2
import unittest


def process_image(state, input_width, input_height):
  image = Image.fromarray(state).crop([8, 32, 152, 210])
  #  image = image.resize([68, 65]).convert('L')
  image = image.resize([input_width, input_height],
    Image.NEAREST).convert('L')
  img = np.expand_dims(np.array(image, dtype=np.uint8), axis=2)
  return img


class SkipFrameEnvironment(object):
  def __init__(self, name, skip, image_width, image_height):
    self.skip = skip
    self.image_width = image_width
    self.image_height = image_height
    self.env = gym.make(name)
    self.action_size = self.env.action_space.n
    self.states = []
    self.lives = 0

  def reset(self):
    states = deque(maxlen=self.skip)
    if self.lives == 0:
      state = self.env.reset()
      states.append(process_image(state, self.image_width, self.image_height))

    end = False
    noop = 0
    while len(states) < self.skip:
      state, _, done, info = self.env.step(noop)
      states.append(process_image(state, self.image_width, self.image_height))
      end |= done
      self.lives = info['ale.lives']
    return np.concatenate(states, axis=2)

  def step(self, action):
    states = deque(maxlen=self.skip)
    R = 0
    end = False
    for i in range(self.skip):
      state, reward, done, info = self.env.step(action)
      states.append(process_image(state, self.image_width, self.image_height))
      R += reward
      end |= done

      if info['ale.lives'] < self.lives:
        end |= True
      self.lives = info['ale.lives']
    return np.concatenate(states, axis=2), R, end, info['ale.lives']

  def render(self):
    self.env.render()


class HistoryFrameEnvironment(object):
  def __init__(self, name, history_size, image_width, image_height,
      record=False, directory=None):
    self.history = deque(maxlen=history_size)
    self.image_width = image_width
    self.image_height = image_height
    self.env = gym.make(name)
    self.action_size = self.env.action_space.n
    self.lives = 0
    self.history_size = history_size

    if record:
      self.env = gym.wrappers.Monitor(self.env, directory,
        video_callable=lambda episode_id: True)

  def reset(self):
    if self.lives == 0:
      self.state = self.env.reset()
      # start the game
    self.state, _, _, info = self.env.step(1)
    self.lives = info['ale.lives']
    self.state = process_image(self.state, self.image_width, self.image_height)

    for _ in range(self.history_size):
      self.history.append(self.state)
    return np.concatenate(self.history, axis=2)

  def step(self, action):
    self.state, reward, done, info = self.env.step(action)
    self.state = process_image(self.state, self.image_width, self.image_height)
    self.history.append(self.state)
    if info['ale.lives'] < self.lives:
      done = True
      reward = -1
    self.lives = info['ale.lives']
    return np.concatenate(self.history, axis=2), reward, done, self.lives

  def render(self):
    self.env.render()


class SimpleEnvironment(object):
  def __init__(self, name):
    self.env = gym.make(name)
    self.action_size = self.env.action_space.n
    self.lives = 0

  def reset(self):
    if self.lives == 0:
      self.state = self.env.reset()
    # start the game
    self.state, _, _, info = self.env.step(1)
    self.lives = info['ale.lives']
    return self.state

  def step(self, action):
    self.state, reward, done, info = self.env.step(action)
    if info['ale.lives'] < self.lives:
      done = True
      reward = -1
    self.lives = info['ale.lives']
    return self.state, reward, done, info['ale.lives']

  def render(self):
    self.env.render()


class TestEnvironment(unittest.TestCase):
  def setUp(self):
    self.run_count = 100
    self.render = True

  def test_skipframe_env(self):
    env = SkipFrameEnvironment('BreakoutNoFrameskip-v0', 4, 84, 84)

    for i in range(self.run_count):
      state = env.reset()
      self.assertEqual(state.shape[2], 4)
      steps = 0

      total_reward = 0
      while True:
        action = randint(0, 3)
        next_state, reward, done, lives = env.step(action)

        self.assertEqual(state.shape[2], 4)

        steps += 1
        total_reward += reward

        if self.render:
          env.render()

        if done:
          break

  def test_history_env(self):
    env = HistoryFrameEnvironment('BreakoutDeterministic-v0', 4, 84, 84)

    for i in range(self.run_count):
      state = env.reset()
      self.assertEqual(state.shape[2], 4)
      steps = 0

      total_reward = 0
      while True:
        action = randint(0, 3)
        next_state, reward, done, lives = env.step(action)

        self.assertEqual(state.shape[2], 4)

        steps += 1
        total_reward += reward

        eq = np.equal(next_state[:, :, 0:3], state[:, :, 1:]).all()
        self.assertTrue(eq)

        state = next_state

        if self.render:
          env.render()

        if done:
          break

  def test_simple_env(self):
    env = SimpleEnvironment('BreakoutDeterministic-v0')

    for i in range(self.run_count):
      state = env.reset()
      env.step(1)

      self.assertEqual(state.shape[0], 210)
      self.assertEqual(state.shape[1], 160)
      self.assertEqual(state.shape[2], 3)
      steps = 0

      total_reward = 0
      while True:
        #  action = randint(0, 3)
        action = 0
        next_state, reward, done, lives = env.step(action)

        self.assertEqual(next_state.shape[0], 210)
        self.assertEqual(next_state.shape[1], 160)
        self.assertEqual(next_state.shape[2], 3)

        steps += 1
        total_reward += reward

        if self.render:
          env.render()

        #  time.sleep(0.1)

        if done:
          break


def main():
  unittest.main()


if __name__ == '__main__':
  main()

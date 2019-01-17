from time import sleep

from gym import Wrapper
from gym import make
from gym.spaces import Box
from PIL import Image
import numpy as np


class LossLifeDone(Wrapper):
  def __init__(self, env):
    super(LossLifeDone, self).__init__(env)
    self._life = None
    self._last_state = None

  def reset(self):
    if self._life == 0 or self._life is None:
      return self.env.reset()
    else:
      return self._last_state

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    life = info['ale.lives']
    if self._life is None:
      self._life = life
    if self._life != life:
      self._life = life
      done = True
    self._last_state = state
    return state, reward, done, info


class FireAtStart(Wrapper):
  def __init__(self, env):
    super(FireAtStart, self).__init__(env)
    self._fire = True

  def reset(self):
    self._fire = True
    return self.env.reset()

  def step(self, action):
    if self._fire:
      action = 1
      self._fire = False
    state, reward, done, info = self.env.step(action)
    return state, reward, done, info


class SkipObservation(Wrapper):
  def __init__(self, env, skip=1):
    super(SkipObservation, self).__init__(env)
    self.skip = skip

  def reset(self):
    return self.env.reset()

  def step(self, action):
    total_reward = 0
    total_done = False
    for _ in range(self.skip):
      state, reward, done, info = self.env.step(action)
      total_reward += reward
      total_done = total_done or done
    return state, total_reward, total_done, info


class GrayObservation(Wrapper):
  def __init__(self, env):
    super(GrayObservation, self).__init__(env)
    shape = self.env.observation_space.shape
    self.observation_space = Box(0, 255, shape[:-1], np.uint8)

  def _gray(self, state):
    r, g, b = state[:,:,0], state[:,:,1], state[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

  def reset(self):
    return self._gray(self.env.reset())

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    return self._gray(state), reward, done, info


class CroppedObservation(Wrapper):
  def __init__(self, env, x1=None, y1=None, x2=None, y2=None):
    super(CroppedObservation, self).__init__(env)
    self.x1 = x1 if x1 is not None else 0
    self.y1 = y1 if y1 is not None else 0
    self.x2 = x2 if x2 is not None else self.env.observation_space.shape[1]
    self.y2 = y2 if y2 is not None else self.env.observation_space.shape[0]
    w = self.x2 - self.x1
    h = self.y2 - self.y1
    c = self.env.observation_space.shape[2]
    self.observation_space = Box(0, 255, [h, w, c], dtype=np.uint8)

  def _crop(self, state):
    return state[self.y1:self.y2, self.x1:self.x2, ...]

  def reset(self):
    return self._crop(self.env.reset())

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    return self._crop(state), reward, done, info


class ResizeObservation(Wrapper):
  def __init__(self, env, w=None, h=None):
    super(ResizeObservation, self).__init__(env)
    self.w = w
    self.h = h
    if self.w is not None and self.h is not None:
      self.observation_space = Box(0, 255, [self.h, self.w], np.uint8)

  def _resize(self, state):
    if self.w is not None and self.h is not None:
      img = Image.fromarray(state)
      img = img.resize([self.w, self.h], Image.ANTIALIAS)
      return np.array(img)
    return state

  def reset(self):
    return self._resize(self.env.reset())

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    return self._resize(state), reward, done, info


def wrap_env(env, skip=4, input_size=84):
  env = LossLifeDone(env)
  env = FireAtStart(env)
  env = SkipObservation(env, skip)
  env = CroppedObservation(env, 8, 32, 152, 210)
  env = GrayObservation(env)
  env = ResizeObservation(env, input_size, input_size)
  return env


def get_breakout_env():
  env = make('Breakout-v0')
  return wrap_env(env)


def test():
  env = get_breakout_env()

  for episode in range(10):
    state = env.reset()
    assert state is not None

    total_reward = 0
    while True:
      state, reward, done, info = env.step(env.action_space.sample())
      #  state, reward, done, info = env.step(0)
      total_reward += reward
      env.render()
      sleep(0.01)
      if done:
        break
    print('%d. total reward: %f' % (episode, total_reward))
    #  img = Image.fromarray(state)
    #  img.show()


if __name__ == '__main__':
  test()

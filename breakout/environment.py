import gym
import numpy as np
from random import randint
from PIL import Image
from collections import deque
import time
import cv2


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
  def __init__(self, name, history_size, image_width, image_height):
    self.history = deque(maxlen=history_size)
    self.image_width = image_width
    self.image_height = image_height
    self.env = gym.make(name)
    self.action_size = self.env.action_space.n
    self.states = []
    self.lives = 0
    self.history_size = history_size

  def reset(self):
    if self.lives == 0:
      for _ in range(self.history_size - 1):
        self.history.append(
          np.zeros([self.image_width, self.image_height, 1], dtype=np.uint8))

      state = self.env.reset()
      self.history.append(process_image(
        state, self.image_width, self.image_height))
    else:
      noop = 0
      state, _, _, info = self.env.step(noop)
      self.lives = info['ale.lives']
    return np.concatenate(self.history, axis=2)

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    self.history.append(process_image(
      state, self.image_width, self.image_height))
    done = info['ale.lives'] < self.lives
    self.lives = info['ale.lives']
    return np.concatenate(self.history, axis=2), \
      reward, done, info['ale.lives']

  def render(self):
    self.env.render()



def main():
  #  env = SkipFrameEnvironment('BreakoutNoFrameskip-v4', 4, 84, 84)
  env = HistoryFrameEnvironment('BreakoutDeterministic-v4', 4, 84, 84)

  for i in range(100):
    state = env.reset()
    assert state.shape[2] == 4
    steps = 0

    total_reward = 0
    while True:
      action = randint(0, 3)
      state, reward, done, lives = env.step(action)
      assert state.shape[2] == 4
      steps += 1
      total_reward += reward

      env.render()

      if done:
        break

    print('steps: %d' % steps)
    print('total reward: %f' % (total_reward))


if __name__ == '__main__':
  main()

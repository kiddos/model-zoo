import unittest
import numpy as np
import random
import gym
import time
from collections import deque
from PIL import Image

from environment import SimpleEnvironment


class ReplayBuffer(object):
  def __init__(self, replay_buffer_size, image_width, image_height,
      history_size):
    self.w, self.h = image_width, image_height
    self.size = replay_buffer_size
    self.history_size = history_size
    self.history = deque(maxlen=history_size)
    self._state = np.zeros(shape=[replay_buffer_size, image_height,
      image_width], dtype=np.uint8)
    self._action = np.zeros(shape=[replay_buffer_size], dtype=np.int32)
    self._reward = np.zeros(shape=[replay_buffer_size], dtype=np.float32)
    self._done = np.zeros(shape=[replay_buffer_size], dtype=np.bool)
    self._current_index = 0
    self._current_size = 0

  def process_image(self, state):
    image = Image.fromarray(state).crop([8, 32, 152, 210])
    image = image.resize([self.w, self.h], Image.NEAREST).convert('L')
    img = np.array(image, dtype=np.uint8)
    return img

  def add(self, next_state, action, reward, done):
    next_state = self.process_image(next_state)
    self.history.append(next_state)

    self._state[self._current_index, ...] = next_state
    self._action[self._current_index] = action
    self._reward[self._current_index] = reward
    self._done[self._current_index] = done

    self._current_index = (self._current_index + 1) % self.size
    self._current_size = min(self._current_size + 1, self.size)

  def add_init_state(self, state):
    padd = np.zeros(shape=[self.h, self.w])
    for _ in range(self.history_size - 1):
      self.history.append(padd)

    self.add(state, 0, 0, False)

  @property
  def current_size(self):
    return self._current_size

  def get_state(self, index):
    index_from = index - self.history_size + 1
    index_to = index + 1
    state = self._state[index_from:index_to, ...]
    return np.transpose(state, (1, 2, 0))

  def sample(self, batch_size):
    states = []
    actions = []
    next_states = []
    rewards = []
    done = []
    min_index = self.history_size
    if self._current_index < self.history_size:
      min_index *= 2
    max_index = self._current_size - 1
    for b in range(batch_size):
      index = random.randint(min_index, max_index)
      while self._done[index - 1]:
        index = random.randint(min_index, max_index)

      state = np.copy(self.get_state(index - 1))
      action = self._action[index]
      next_state = np.copy(self.get_state(index))
      over = self._done[index]
      reward = self._reward[index]

      # padd zero
      game_over = self._done[(index - self.history_size):index]
      for i in range(self.history_size - 1, -1, -1):
        if game_over[i]:
          state[:, :, :(i + 1)] = 0
          next_state[:, :, :i] = 0
          break

      states.append(state)
      actions.append(action)
      next_states.append(next_state)
      rewards.append(reward)
      done.append(over)

    states = np.stack(states, axis=0)
    actions = np.array(actions)
    next_states = np.stack(next_states, axis=0)
    rewards = np.array(rewards)
    done = np.array(done)
    return states, actions, next_states, rewards, done

  def last_state(self):
    last_state = np.array(self.history, np.uint8)
    return np.transpose(last_state, (1, 2, 0))


class TestReplayBuffer(unittest.TestCase):
  def setUp(self):
    self.replay_buffer = ReplayBuffer(300, 84, 84, 4)
    self.env = SimpleEnvironment('Breakout-v0')

  def test_add_states(self):
    state = self.env.reset()
    self.replay_buffer.add_init_state(state)
    while True:
      action = self.env.sample_action()
      next_state, reward, done, info = self.env.step(action)
      self.replay_buffer.add(next_state, action, reward, done)
      state = next_state
      if done: break

  def test_sample(self):
    try:
      import cv2
    except:
      raise Exception

    for _ in range(10):
      self.test_add_states()

    states, actions, next_states, rewards, done = \
      self.replay_buffer.sample(300)

    self.assertEqual(states.dtype, np.uint8)
    self.assertEqual(next_states.dtype, np.uint8)
    self.assertEqual(actions.dtype, np.int32)
    self.assertEqual(rewards.dtype, np.float32)
    self.assertEqual(done.dtype, np.bool)

    eq = np.all(states[:, :, :, 1:] == next_states[:, :, :, :3])
    self.assertTrue(eq)

    self.assertEqual(len(states), len(next_states))

    for i in range(len(states)):
      display = np.zeros([84 * 2, 84 * 4, 1], np.uint8)

      for j in range(4):
        display[0:84, (j * 84):((j + 1) * 84), 0] = states[i, :, :, j]
        display[84:, (j * 84):((j + 1) * 84), 0] = next_states[i, :, :, j]
        cv2.line(display, (84 * j, 0), (84 * j, 168), (255, 255, 255), 1)

      ACTION_MEANING = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
      action_text = 'action: %s(%d)' % (ACTION_MEANING[actions[i]], actions[i])
      cv2.putText(display, action_text, (0, 10),
        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
      done_text = 'done: ' + str(done[i])
      cv2.putText(display, done_text, (0, 30),
        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

      done_text = 'reward: ' + str(rewards[i])
      cv2.putText(display, done_text, (0, 50),
        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

      cv2.imshow('States', display)
      key = cv2.waitKey(0)
      if key in [ord('q'), ord('Q')]: break


      #  p1 = Image.fromarray(states[0, :, :, 0])
      #  p2 = Image.fromarray(next_states[0, :, :, 0])
      #  p1.save('p1.png')
      #  p2.save('p2.png')

  def test_last_state(self):
    try:
      import cv2
    except:
      raise Exception

    state = self.env.reset()
    self.replay_buffer.add_init_state(state)
    while True:
      action = self.env.sample_action()
      next_state, reward, done, info = self.env.step(action)

      s = self.replay_buffer.process_image(state)
      last_state = self.replay_buffer.last_state()
      self.assertEqual(s.shape, (84, 84))
      self.assertEqual(last_state.shape, (84, 84, 4))

      eq = np.all(last_state[:, :, -1] == s)
      self.assertTrue(eq)

      self.replay_buffer.add(next_state, action, reward, done)

      display = np.zeros([84, 84 * 4, 1], np.uint8)
      last_state = self.replay_buffer.last_state()

      for j in range(4):
        display[:, (j * 84):((j + 1) * 84), 0] = last_state[:, :, j]
        cv2.line(display, (84 * j, 0), (84 * j, 84), (255, 255, 255), 1)

      ACTION_MEANING = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
      action_text = 'action: %s(%d)' % (ACTION_MEANING[action], action)
      cv2.putText(display, action_text, (200, 10),
        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

      done_text = 'done: ' + str(done)
      cv2.putText(display, done_text, (230, 30),
        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

      done_text = 'reward: ' + str(reward)
      cv2.putText(display, done_text, (230, 50),
        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

      cv2.imshow('States', display)
      key = cv2.waitKey(1000)
      if key in [ord('q'), ord('Q')]: break

      state = next_state
      if done: break


def main():
  unittest.main()


if __name__ == '__main__':
  main()

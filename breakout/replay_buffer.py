import unittest
import numpy as np
import random
import gym
import time
from collections import deque
from PIL import Image

from environment import get_training_env


class ReplayBuffer(object):
  def __init__(self, replay_buffer_size, image_width, image_height,
      history_size):
    self.w, self.h = image_width, image_height
    self.size = replay_buffer_size
    self.history_size = history_size
    self.history = deque(maxlen=history_size - 1)
    self._state = np.zeros(shape=[replay_buffer_size, image_height,
      image_width], dtype=np.uint8)
    self._action = np.zeros(shape=[replay_buffer_size], dtype=np.int32)
    self._reward = np.zeros(shape=[replay_buffer_size], dtype=np.float32)
    self._done = np.zeros(shape=[replay_buffer_size], dtype=np.bool)
    self._current_index = 0
    self._current_size = 0

  def _add(self, index, state, action, reward, done):
    self._state[index] = state
    self._reward[index] = reward
    self._action[index] = action
    self._done[index] = done

  def add(self, state, action, reward, done):
    self._add(self._current_index, state, action, reward, done)
    self._current_index = (self._current_index + 1) % self.size
    self._current_size = min(self._current_size + 1, self.size)

    if done:
      self.history.clear()
    else:
      self.history.append(state)

  def recent_state(self, latest_state):
    recent = list(self.history)
    states = [np.zeros([self.h, self.w], np.uint8)] * \
      (self.history.maxlen - len(recent))
    states.extend([state for state in recent])
    states.append(latest_state)
    return np.stack(states, axis=2)

  @property
  def current_size(self):
    return self._current_size

  def _slice(self, data, start, end):
    a1 = data[start:]
    a2 = data[:end]
    return np.concatenate((a1, a2), axis=0)

  def _pad(self, state, reward, action, done):
    for k in range(self.history_size - 2, -1, -1):
      if done[k]:
        state = np.copy(state)
        state[:k + 1].fill(0)
        break
    state = state.transpose(1, 2, 0)
    return state[:, :, 0:self.history_size], action[-2], \
      state[:, :, 1:], reward[-2], done[-2]

  def _sample(self, index):
    index = (self._current_index + index) % self._current_size
    k = self.history_size + 1

    if index + k <= self._current_size:
      state = self._state[index:(index + k)]
      reward = self._reward[index:(index + k)]
      action = self._action[index:(index + k)]
      done = self._done[index:(index + k)]
    else:
      end = index + k - self._current_size
      state = self._slice(self._state, index, end)
      reward = self._slice(self._reward, index, end)
      action = self._slice(self._action, index, end)
      done = self._slice(self._done, index, end)
    sampled = self._pad(state, reward, action, done)
    return sampled

  def _process_batch(self, batch):
    states, actions, next_states, rewards, done = batch
    states = np.asarray(states, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.int8)
    next_states = np.asarray(next_states, dtype=np.uint8)
    rewards = np.asarray(rewards, dtype=np.float32)
    done = np.asarray(done, dtype=np.bool)
    return states, actions, next_states, rewards, done

  def sample(self, batch_size):
    indices = np.random.randint(0, self._current_size - self.history_size - 1,
      [batch_size])
    batch = zip(*[self._sample(i) for i in indices])
    return self._process_batch(batch)


class TestReplayBuffer(unittest.TestCase):
  def setUp(self):
    self.replay_buffer = ReplayBuffer(300, 84, 84, 4)
    self.env = get_training_env('Breakout-v0', 84, 84)

  def test_add_states(self):
    state = self.env.reset()

    while True:
      action = self.env.action_space.sample()
      next_state, reward, done, info = self.env.step(action)
      self.replay_buffer.add(state, action, reward, done)
      state = next_state
      if done:
        self.env.reset()
        break

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
    self.assertEqual(actions.dtype, np.int8)
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

    for episode in range(5):
      state = self.env.reset()
      while True:
        action = self.env.action_space.sample()
        next_state, reward, done, info = self.env.step(action)

        last_state = self.replay_buffer.recent_state(state)
        self.assertEqual(last_state.shape, (84, 84, 4))

        eq = np.all(last_state[:, :, -1] == state)
        self.assertTrue(eq)

        self.replay_buffer.add(state, action, reward, done)
        state = next_state

        display = np.zeros([84, 84 * 4, 1], np.uint8)
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
        if key in [ord('q'), ord('Q')]:
          self.env.reset()
          break

        if done:
          self.env.reset()
          break


def main():
  unittest.main()


if __name__ == '__main__':
  main()

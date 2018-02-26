import unittest
import numpy as np
import random
import gym
from collections import deque
from PIL import Image


class ReplayBuffer(object):
  def __init__(self, replay_buffer_size, image_width, image_height,
      history_size):
    self.w, self.h = image_width, image_height
    self.size = replay_buffer_size
    self.history_size = history_size
    self.history = deque(maxlen=history_size)
    self._state = np.zeros(shape=[replay_buffer_size, image_height,
      image_width], dtype=np.uint8)
    self._action = np.zeros(shape=[replay_buffer_size], dtype=np.int16)
    self._reward = np.zeros(shape=[replay_buffer_size], dtype=np.int16)
    self._done = np.zeros(shape=[replay_buffer_size], dtype=np.bool)
    self._current_index = 0
    self._current_size = 0

    padd = np.zeros(shape=[self.h, self.w])
    for _ in range(self.history_size - 1):
      self.history.append(padd)

  def process_image(self, state):
    image = Image.fromarray(state).crop([8, 32, 152, 210])
    image = image.resize([self.w, self.h], Image.NEAREST).convert('L')
    img = np.array(image, dtype=np.uint8)
    return img

  def add(self, next_state, action, reward, done):
    next_state = self.process_image(next_state)
    self.history.append(next_state)

    self._state[self._current_index, ...] = next_state
    self._action[self._current_index] = np.array(action, np.int16)
    self._reward[self._current_index] = np.sign(reward).astype(np.int16)
    self._done[self._current_index] = np.array(done, np.bool)

    self._current_index = (self._current_index + 1) % self.size
    self._current_size = min(self._current_size + 1, self.size)

  def add_init_state(self, state):
    self.add(state, 0, 0, False)

  @property
  def current_size(self):
    return len(self._state)

  def get_state(self, index):
    index_from = index - self.history_size
    index_to = index
    state = self._state[index_from:index_to, ...]
    return np.transpose(state, (1, 2, 0))

  def terminal(self, index):
    return self._done[index - self.history_size:index].any()

  def sample(self, batch_size):
    states = []
    actions = []
    next_states = []
    rewards = []
    done = []
    min_index = self.history_size
    if self._current_index < self.history_size:
      min_index *= 2
    for b in range(batch_size):
      index = random.randint(min_index, self._current_size - 1)
      while self._done[index - 1]:
        index = random.randint(min_index, self._current_size - 1)

      state = self.get_state(index)
      action = self._action[index]
      next_state = self.get_state(index + 1)
      game_over = self._done[(index - self.history_size):index]
      reward = self._reward[index]
      for i in range(-2, -self.history_size - 1, -1):
        if game_over[i]:
          state[0:i, ...] = 0
          next_state[0:(i - 1), ...] = 0
          break

      states.append(state)
      actions.append(action)
      next_states.append(next_state)
      rewards.append(reward)
      done.append(self._done[index - 1])

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
    self.replay_buffer = ReplayBuffer(10, 84, 84, 4)
    self.env = gym.make('Breakout-v0')

  def test_add_states(self):
    state = self.env.reset()
    self.replay_buffer.add(state, 0, 0, False)
    while True:
      action = random.randint(0, 3)
      next_state, reward, done, info = self.env.step(action)
      self.replay_buffer.add(next_state, action, reward, done)
      state = next_state
      if done: break

  def test_sample(self):
    self.test_add_states()
    states, actions, next_states, rewards, done = self.replay_buffer.sample(32)

    self.assertEqual(states.dtype, np.uint8)
    self.assertEqual(next_states.dtype, np.uint8)
    self.assertEqual(actions.dtype, np.int16)
    self.assertEqual(rewards.dtype, np.int16)
    self.assertEqual(done.dtype, np.bool)

    eq = np.all(states[:, :, :, 1:] == next_states[:, :, :, :3])
    self.assertTrue(eq)

    #  p1 = Image.fromarray(states[0, :, :, 0])
    #  p2 = Image.fromarray(next_states[0, :, :, 0])
    #  p1.save('p1.png')
    #  p2.save('p2.png')

  def test_last_state(self):
    state = self.env.reset()
    self.replay_buffer.add(state, 0, 0, False)
    while True:
      action = random.randint(0, 3)
      next_state, reward, done, info = self.env.step(action)

      s = self.replay_buffer.process_image(state)
      last_state = self.replay_buffer.last_state()
      self.assertEqual(s.shape, (84, 84))
      self.assertEqual(last_state.shape, (84, 84, 4))

      eq = np.all(last_state[:, :, -1] == s)
      self.assertTrue(eq)

      self.replay_buffer.add(next_state, action, reward, done)
      state = next_state
      if done: break

  def test_multiple_sample(self):
    for _ in range(20):
      self.test_sample()

  def test_multiple_last_state(self):
    for _ in range(20):
      self.test_last_state()


def main():
  unittest.main()


if __name__ == '__main__':
  main()

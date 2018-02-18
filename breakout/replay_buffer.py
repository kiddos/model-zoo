import unittest
import numpy as np
import random
import gym
from PIL import Image


class ReplayBuffer(object):
  def __init__(self, replay_buffer_size, image_width, image_height,
      history_size):
    self.size = replay_buffer_size
    self.history_size = history_size
    self._state = np.zeros(
      shape=[replay_buffer_size, image_height, image_width],
      dtype=np.uint8)
    self._action = np.zeros(shape=[replay_buffer_size - 1], dtype=np.int16)
    self._reward = np.zeros(shape=[replay_buffer_size - 1], dtype=np.int16)
    self._done = np.zeros(shape=[replay_buffer_size - 1], dtype=np.bool)
    self._current_size = 0
    self._current_index = 0
    self.padd()

  def padd(self):
    empty = np.zeros(shape=[84, 84, 3], dtype=np.uint8)
    no_op = 0
    for _ in range(self.history_size - 1):
      self.add(empty, no_op, 0, False)

  def process_image(self, state):
    w, h = self._state.shape[1:3]
    image = Image.fromarray(state).crop([8, 32, 152, 210])
    image = image.resize([w, h], Image.NEAREST).convert('L')
    img = np.array(image, dtype=np.uint8)
    return img

  def init_state(self, state):
    self._state[self._current_index, ...] = self.process_image(state)

  def add(self, next_state, action, reward, done):
    self._state[self._current_index + 1, ...] = self.process_image(next_state)
    self._action[self._current_index] = action
    self._reward[self._current_index] = np.sign(reward)
    self._done[self._current_index] = done
    self._current_index = (self._current_index + 1) % self.size
    self._current_size = min(self._current_size + 1, self.size - 1)

  @property
  def current_size(self):
    return self._current_size

  def get_state(self, index):
    return np.transpose(
        self._state[(index - self.history_size + 1):(index + 1), ...], (1, 2, 0))

  def sample(self, batch_size):
    states = []
    actions = []
    next_states = []
    rewards = []
    done = []
    for b in range(batch_size):
      while True:
        index = random.randint(self.history_size, self._current_size - 1)
        if not self._done[(index - self.history_size):index].any():
          states.append(self.get_state(index - 1))
          actions.append(self._action[index])
          next_states.append(self.get_state(index))
          rewards.append(self._reward[index])
          done.append(self._done[index])
          break
    states = np.stack(states, axis=0)
    actions = np.array(actions)
    next_states = np.stack(next_states, axis=0)
    rewards = np.array(rewards)
    done = np.array(done)
    return states, actions, next_states, rewards, done

  def last_state(self):
    return self.get_state(self._current_index - 1)


class TestReplayBuffer(unittest.TestCase):
  def setUp(self):
    self.replay_buffer = ReplayBuffer(1000000, 84, 84, 4)
    self.env = gym.make('Breakout-v0')

  def test_add_states(self):
    state = self.env.reset()
    self.replay_buffer.init_state(state)
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
    self.replay_buffer.init_state(state)
    while True:
      action = random.randint(0, 3)
      next_state, reward, done, info = self.env.step(action)
      self.replay_buffer.add(next_state, action, reward, done)

      s = self.replay_buffer.process_image(state)
      last_state = self.replay_buffer.last_state()
      eq = np.all(last_state[:, :, -1] == s)
      self.assertTrue(eq)
      state = next_state
      if done: break

  def test_multiple_sample(self):
    for _ in range(20):
      self.test_sample()


def main():
  unittest.main()


if __name__ == '__main__':
  main()

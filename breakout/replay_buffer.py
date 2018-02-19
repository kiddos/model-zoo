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
    self._state = deque(maxlen=replay_buffer_size)
    self._action = deque(maxlen=replay_buffer_size - 1)
    self._reward = deque(maxlen=replay_buffer_size - 1)
    self._done = deque(maxlen=replay_buffer_size - 1)
    self.padd()

  def padd(self):
    empty = np.zeros(shape=[84, 84, 3], dtype=np.uint8)
    no_op = 0
    for _ in range(self.history_size - 1):
      self.add(empty, no_op, 0, False)

  def process_image(self, state):
    image = Image.fromarray(state).crop([8, 32, 152, 210])
    image = image.resize([self.w, self.h], Image.NEAREST).convert('L')
    img = np.array(image, dtype=np.uint8)
    return img

  def init_state(self, state):
    self._state.append(self.process_image(state))

  def add(self, next_state, action, reward, done):
    self._state.append(self.process_image(next_state))
    self._action.append(np.array(action, dtype=np.int16))
    self._reward.append(np.sign(np.array(reward, dtype=np.int16)))
    self._done.append(done)

  @property
  def current_size(self):
    return len(self._state)

  def get_state(self, index):
    state = []
    for i in range(index - self.history_size + 1, index + 1):
      state.append(self._state[i])
    state = np.stack(state, axis=0)
    return np.transpose(state, (1, 2, 0))

  def terminal(self, index):
    term = False
    for i in range(index - self.history_size, index):
      term |= self._done[i]
    return term

  def sample(self, batch_size):
    states = []
    actions = []
    next_states = []
    rewards = []
    done = []
    current_size = len(self._done)
    for b in range(batch_size):
      while True:
        index = random.randint(self.history_size, current_size - 1)
        if not self.terminal(index):
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
    return self.get_state(-2)


class TestReplayBuffer(unittest.TestCase):
  def setUp(self):
    self.replay_buffer = ReplayBuffer(10, 84, 84, 4)
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
      self.assertEqual(s.shape, (84, 84))
      self.assertEqual(last_state.shape, (84, 84, 4))

      eq = np.all(last_state[:, :, -1] == s)
      self.assertTrue(eq)
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

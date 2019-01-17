from time import sleep
import numpy as np
from PIL import Image

from gym_wrapper import get_breakout_env


class ReplayBuffer(object):
  HISTORY_SIZE = 4

  def __init__(self, buffer_size, w, h):
    self.buffer_size = buffer_size
    self.states = np.zeros(shape=[buffer_size, h, w], dtype=np.uint8)
    self.actions = np.zeros(shape=[buffer_size], dtype=np.int32)
    self.dones = np.zeros(shape=[buffer_size], dtype=np.bool)
    self.rewards = np.zeros(shape=[buffer_size], dtype=np.float32)
    self.current_index = 0
    self.current_size = 0

  def add(self, state, action, reward, done):
    i = self.current_index
    self.states[i, ...] = state
    self.actions[i] = action
    self.dones[i] = done
    self.rewards[i] = reward

    self.current_index = (self.current_index + 1) % (self.buffer_size)
    self.current_size = min(self.buffer_size, self.current_size + 1)

  def sample(self):
    if not self.ready():
      return None

    i = np.random.randint(self.HISTORY_SIZE, self.current_size - 1)
    a = self.actions[i]
    d = self.dones[i]
    r = self.rewards[i]

    s = [self.states[i]]
    p = False
    for n in range(i - 1, i - self.HISTORY_SIZE - 1, -1):
      p = p or self.dones[n]
      if p:
        s = [np.zeros_like(s[0])] + s
      else:
        s = [self.states[n, ...]] + s
    s = np.stack(s, axis=2)
    return s[:, :, :self.HISTORY_SIZE], s[:, :, 1:], a, r, d

  def ready(self):
    return self.current_size >= self.HISTORY_SIZE

  def next(self, batch_size):
    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []
    for _ in range(batch_size):
      s, ns, a, r, d = self.sample()
      states.append(s)
      next_states.append(ns)
      actions.append(a)
      rewards.append(r)
      dones.append(d)
    return np.array(states), np.array(next_states), np.array(actions), \
        np.array(rewards), np.array(dones)


def test():
  replay_buffer = ReplayBuffer(100, 84, 84)
  env = get_breakout_env()

  for episode in range(10):
    state = env.reset()
    assert state is not None

    total_reward = 0
    while True:
      action = env.action_space.sample()
      next_state, reward, done, info = env.step(action)
      replay_buffer.add(state, action, reward, done)
      total_reward += reward
      state = next_state
      if done:
        break

    states, next_states, actions, rewards, dones = replay_buffer.sample()
    #  print(states.shape)
  data = replay_buffer.next(10)
  print(data[0].shape)

  display = np.zeros(shape=[84 * 2, 84 * 4], dtype=np.uint8)
  for i in range(4):
    display[:84, 84*i:84*(i+1)] = states[:, :, i]
    display[84:, 84*i:84*(i+1)] = next_states[:, :, i]

  img = Image.fromarray(display)
  img.show()


if __name__ == '__main__':
  test()

import gym
import numpy as np
from random import randint
from PIL import Image


class SkipFrameEnvironment(object):
  def __init__(self, name, skip, image_width, image_height):
    self.skip = skip
    self.image_width = image_width
    self.image_height = image_height
    self.env = gym.make(name)
    self.action_size = self.env.action_space.n
    self.states = []

  def reset(self):
    state = self.env.reset()
    R = 0
    end = False
    states = [self.process_image(state)]
    for i in range(self.skip - 1):
      state, reward, done, _ = self.env.step(randint(0, self.action_size - 1))
      R += reward
      states.append(self.process_image(state))
      end |= done

    if end:
      return self.reset()
    else:
      return np.concatenate(states, axis=2)

  def step(self, action):
    states = []
    R = 0
    end = False
    for i in range(self.skip):
      state, reward, done, info = self.env.step(action)
      states.append(self.process_image(state))
      R += reward
      end |= done
    return np.concatenate(states, axis=2), R, end

  def process_image(self, state):
    image = Image.fromarray(state).crop([8, 32, 152, 210])
    #  image = image.resize([68, 65]).convert('L')
    image = image.resize([self.image_width, self.image_height],
      Image.NEAREST).convert('L')
    img = np.expand_dims(np.array(image, dtype=np.uint8), axis=2)
    return img

  def render(self):
    self.env.render()


def main():
  env = SkipFrameEnvironment('BreakoutNoFrameskip-v4', 4, 84, 84)
  state = env.reset()
  steps = 0
  total_reward = 0
  while True:
    action = randint(0, 3)
    state, reward, done = env.step(action)
    steps += 1
    total_reward += reward

    env.render()

    if done:
      break

  print('steps: %d' % steps)
  print('total reward: %f' % (total_reward))


if __name__ == '__main__':
  main()

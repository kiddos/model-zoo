import gym

env = gym.make('BreakoutDeterministic-v4')
env.reset()
while True:
  state, r, done, info = env.step(env.action_space.sample())
  env.render()
  if done:
    break

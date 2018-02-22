import tensorflow as tf
import numpy as np
import os
import logging
import gym
import time

from environment import HistoryFrameEnvironment


logging.basicConfig()
logger = logging.getLogger('run')
logger.setLevel(logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'breakout.pb', 'model to run')
tf.app.flags.DEFINE_bool('record', False, 'record result')
tf.app.flags.DEFINE_bool('render', True, 'render result')


def load_graph():
  if os.path.isfile(FLAGS.model):
    with tf.gfile.GFile(FLAGS.model, 'rb') as gf:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(gf.read())

      with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        return graph
  else:
    logger.error('model %s not found', FLAGS.model)


def main(_):
  env = HistoryFrameEnvironment('BreakoutDeterministic-v0', 4, 84, 84,
    FLAGS.record, '/tmp/breakout-record')

  graph = load_graph()
  input_state = graph.get_tensor_by_name('import/state:0')
  q_values = graph.get_tensor_by_name('import/train/output/q_values:0')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config, graph=graph) as sess:
    for i in range(100):
      state = env.reset()
      steps = 0
      total_reward = 0
      while True:
        action_prob = sess.run(q_values, feed_dict={
          input_state: np.expand_dims(state, axis=0)
        })
        action = np.argmax(action_prob[0, :])
        state, reward, done, lives = env.step(action)
        assert state.shape[2] == 4
        steps += 1
        total_reward += reward

        if FLAGS.render:
          env.render()

        if done:
          break


if __name__ == '__main__':
  tf.app.run()

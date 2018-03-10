import tensorflow as tf
import numpy as np
import random
import coloredlogs
import logging
import signal
import sys
import os

from experience_replay import ExperienceReplay
from dqn import DQN, DQNConfig

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('saving', False, 'saving model')
tf.app.flags.DEFINE_integer('display_epoch', 200, 'display result per episode')
tf.app.flags.DEFINE_integer('save_epoch', 30000, 'save model per episode')
tf.app.flags.DEFINE_integer('summary_epoch', 10, 'save summary per episode')

# hyperparameters
tf.app.flags.DEFINE_integer('init_replay_buffer_size', 10000,
  'replay buffer starting size')
tf.app.flags.DEFINE_integer('replay_buffer_size', 1000000,
  'replay buffer max size')
tf.app.flags.DEFINE_integer('max_epoches', 5000000,
  'number of episodes to run')
tf.app.flags.DEFINE_integer('update_frequency', 10000,
  'update target network per episode')
tf.app.flags.DEFINE_integer('decay_to_epoch', 1000000,
  'decay epsilon until epoch')
tf.app.flags.DEFINE_float('start_epsilon', 1.0, 'min epsilon to decay to')
tf.app.flags.DEFINE_float('min_epsilon', 0.1, 'min epsilon to decay to')
tf.app.flags.DEFINE_float('gamma', 0.99, 'discount factor')
tf.app.flags.DEFINE_float('learning_rate', 0.00025, 'learning rate to train')
tf.app.flags.DEFINE_float('decay', 0.95,
  'decay factor for next gradients for RMSProp')
tf.app.flags.DEFINE_float('momentum', 0.95, 'squred momentum for RMSProp')
tf.app.flags.DEFINE_float('eps', 0.01, 'eps for avoiding zero for RMSProp')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size to train')
tf.app.flags.DEFINE_integer('image_width', 84, 'input image width')
tf.app.flags.DEFINE_integer('image_height', 84, 'input image height')
tf.app.flags.DEFINE_bool('use_huber', True, 'use huber loss')
tf.app.flags.DEFINE_integer('skip', 4, 'skip frame')
tf.app.flags.DEFINE_integer('history_length', 4, 'history length')


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('breakout')
logger.setLevel(logging.INFO)


def prepare_folder():
  index = 0
  folder = os.path.join('/tmp', 'breakout_%d' % index)
  while os.path.isdir(folder):
    index += 1
    folder = os.path.join('/tmp', 'breakout_%d' % index)
  return folder


class Trainer(object):
  def __init__(self):
    self.experience_replay = ExperienceReplay('BreakoutDeterministic-v0',
      FLAGS.replay_buffer_size,
      84, 84, 4, self.policy, FLAGS.decay_to_epoch)

    config = DQNConfig()
    config.learning_rate = FLAGS.learning_rate
    config.gamma = FLAGS.gamma
    config.decay = FLAGS.decay
    config.momentum = FLAGS.momentum
    config.eps = FLAGS.eps
    config.input_width = FLAGS.image_width
    config.input_height = FLAGS.image_height
    config.skip = FLAGS.skip
    self.dqn = DQN(config, FLAGS.use_huber)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    logger.info('initializing variables...')
    self.sess.run(tf.global_variables_initializer())
    self.update_target()

    self.epoch = 0
    self.decay_epsilon()

  def __del__(self):
    self.sess.close()

  def train_step(self):
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = \
      self.experience_replay.replay_buffer.sample(FLAGS.batch_size)
    self.sess.run(self.dqn.train_ops, feed_dict={
      self.dqn.state: state_batch,
      self.dqn.action: action_batch,
      self.dqn.next_state: next_state_batch,
      self.dqn.reward: reward_batch,
      self.dqn.done: done_batch,
    })

  def train(self):
    if FLAGS.saving:
      folder = prepare_folder()
      saver = tf.train.Saver(max_to_keep=30)
      summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
        tf.get_default_graph())

    def handle_interrupt(sig, frame):
      self.experience_replay.stop()
      logger.info('saving last...')
      saver.save(self.sess, os.path.join(folder, 'breakout'),
        global_step=self.epoch)
      sys.exit()

    signal.signal(signal.SIGINT, handle_interrupt)

    logger.info('adding replay buffer with epsilon: %f', self.epsilon)
    self.experience_replay.init_replay_buffer(FLAGS.init_replay_buffer_size)
    logger.info('replay buffer size: %d',
      self.experience_replay.replay_buffer.current_size)

    self.experience_replay.start()

    for self.epoch in range(FLAGS.max_epoches + 1):
      if self.epoch % FLAGS.update_frequency == 0:
        self.update_target()

      self.train_step()
      self.decay_epsilon()
      self.experience_replay.set_epsilon(self.epsilon)

      if self.epoch % FLAGS.display_epoch == 0:
        loss = self.compute_loss()
        ave_q = self.compute_average_q_values()

        logger.info('%d. eps: %f, ave: %f, max R: %f, Q: %f, loss: %f',
          self.epoch, self.epsilon,
          self.experience_replay.average_reward,
          self.experience_replay.max_reward,
          ave_q, loss)

      if FLAGS.saving and self.epoch % FLAGS.save_epoch == 0:
        saver.save(self.sess, os.path.join(folder, 'breakout'),
          global_step=self.epoch)

      if FLAGS.saving and self.epoch % FLAGS.summary_epoch == 0:
        summary = self.get_summary()
        summary_writer.add_summary(summary, global_step=self.epoch)

    self.experience_replay.stop()

  def update_target(self):
    self.sess.run(self.dqn.copy_ops)

  def policy(self, state):
    action_prob = self.sess.run(self.dqn.q_values, feed_dict={
      self.dqn.state: np.expand_dims(state, axis=0)
    })
    return np.argmax(action_prob[0, :])

  def compute_loss(self):
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = \
      self.experience_replay.replay_buffer.sample(FLAGS.batch_size)
    return self.sess.run(self.dqn.loss, feed_dict={
      self.dqn.state: state_batch,
      self.dqn.next_state: next_state_batch,
      self.dqn.action: action_batch,
      self.dqn.reward: reward_batch,
      self.dqn.done: done_batch,
    })

  def compute_average_q_values(self):
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = \
      self.experience_replay.replay_buffer.sample(FLAGS.batch_size)
    q = self.sess.run(self.dqn.q_values, feed_dict={
      self.dqn.state: state_batch,
    })
    return np.mean(np.max(q, axis=1))

  def get_summary(self):
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = \
      self.experience_replay.replay_buffer.sample(FLAGS.batch_size)
    return self.sess.run(self.dqn.summary, feed_dict={
      self.dqn.state: state_batch,
      self.dqn.next_state: next_state_batch,
      self.dqn.action: action_batch,
      self.dqn.reward: reward_batch,
      self.dqn.done: done_batch,
    })

  def ready(self):
    return self.experience_replay.replay_buffer.current_size >= \
      FLAGS.init_replay_buffer_size

  def decay_epsilon(self):
    self.epsilon = FLAGS.start_epsilon - \
      (FLAGS.start_epsilon - FLAGS.min_epsilon) * self.epoch / \
      FLAGS.decay_to_epoch
    self.epsilon = max(FLAGS.min_epsilon, self.epsilon)


def main(_):
  trainer = Trainer()
  trainer.train()


if __name__ == '__main__':
  tf.app.run()

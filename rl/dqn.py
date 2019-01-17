import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_actions', 4, 'number of actions')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate to train')
tf.app.flags.DEFINE_float('gamma', 0.9, 'gamma')


class DQN(object):
  def __init__(self, input_shape, inference_fn):
    self.input_shape = input_shape

    with tf.name_scope('inputs'):
      self._setup_inputs()

    self.inference = inference_fn
    q = self.inference(self.state, 'dqn')
    o = tf.one_hot(self.action, FLAGS.num_actions)
    mask_q = tf.reduce_sum(o * q, axis=1)
    self.best_action = tf.argmax(q, axis=1)

    next_q = self.inference(self.next_state, 'target_dqn')
    target_q = tf.stop_gradient(tf.reduce_max(next_q, axis=1))
    target = tf.clip_by_value(self.reward, -1, 1) + \
      FLAGS.gamma * tf.cast(tf.logical_not(self.done), tf.float32) * target_q

    with tf.name_scope('copy'):
      v1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'dqn')
      v2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target_dqn')
      self.copy = []
      for i in range(len(v1)):
        self.copy.append(tf.assign(v2[i], v1[i]))

    self.loss = tf.losses.huber_loss(target, mask_q, scope='loss',
      reduction=tf.losses.Reduction.MEAN)
    with tf.name_scope('optimization'):
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

  def _setup_inputs(self):
    self.state = tf.placeholder(dtype=tf.float32,
      shape=[None] + self.input_shape, name='state')
    self.next_state = tf.placeholder(dtype=tf.float32,
      shape=[None] + self.input_shape, name='next_state')
    self.action = tf.placeholder(dtype=tf.int32, shape=[None], name='action')
    self.reward = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
    self.done = tf.placeholder(dtype=tf.bool, shape=[None], name='done')


def atari_model(inputs, name):
  def act(inputs):
    return tf.nn.leaky_relu(inputs, alpha=0.01)

  with tf.variable_scope(name):
    initializer = tf.variance_scaling_initializer()
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=4, kernel_size=8,
        activation_fn=act, padding='SAME', weights_initializer=initializer)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(conv, 64, stride=2, kernel_size=4,
        activation_fn=act, padding='SAME', weights_initializer=initializer)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(conv, 64, stride=1, kernel_size=3,
        activation_fn=act, padding='SAME', weights_initializer=initializer)

    with tf.name_scope('fully_connected'):
      flatten = tf.contrib.layers.flatten(conv)
      fc = tf.contrib.layers.fully_connected(flatten, 512,
        activation_fn=act, weights_initializer=initializer)

    with tf.name_scope('output'):
      w = tf.get_variable(name + 'ow', shape=[512, 4],
        initializer=initializer)
      b = tf.get_variable(name + 'ob', shape=[4],
        initializer=tf.zeros_initializer())
      outputs = tf.add(tf.matmul(fc, w), b, name='q_values')
  return outputs


def test():
  DQN([84, 84, 4], atari_model)


if __name__ == '__main__':
  test()

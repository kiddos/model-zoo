import tensorflow as tf
import numpy as np
import logging
import time
import os


class MNISTConvolutionModel(object):
  def __init__(self, input_width, input_height, input_channel, output_size,
      model_name='MNISTConvolutionModel', learning_rate=1e-3,
      decay=0.9, saving=True):
    # logger
    logging.basicConfig()
    self.logger = logging.getLogger(model_name)
    self.logger.setLevel(logging.INFO)
    self.logger.info('setting up model...')
    # model
    with tf.variable_scope(model_name):
      self.inputs, self.labels, self.keep_prob, \
        self.outputs, self.loss, self.accuracy, self.train_op = \
        self._build_model(input_width, input_height, input_channel,
        output_size, learning_rate, decay)
    # checkpoint
    self.start_epoch = 0
    self.saving = saving
    if saving:
      self.checkpoint_path, summary_path = self._prepare_save_dir(model_name)
      # saver
      self.logger.info('setting up saver...')
      self.saver = tf.train.Saver()
      # summary writer
      self.logger.info('setting up summary writer...')
      self.summary_writer = tf.summary.FileWriter(summary_path,
        tf.get_default_graph())
    self.merged_summary = tf.summary.merge_all()

  def _prepare_save_dir(self, model_name):
    index = 0
    while os.path.isdir(model_name + str(index)):
      index += 1
    model_path = model_name + str(index)
    self.logger.info('creating model directory %s...' % (model_path))
    os.mkdir(model_path)
    summary_path = os.path.join(model_path, 'summary')
    os.mkdir(summary_path)
    checkpoint_path = os.path.join(model_path, model_name)
    return checkpoint_path, summary_path

  def _build_model(self, input_width, input_height, input_channel,
    output_size, learning_rate, decay):
    # inputs
    inputs = tf.placeholder(dtype=tf.float32,
    shape=[None, input_height, input_width, input_channel])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
    keep_prob = tf.placeholder(dtype=tf.float32, shape=())
    tf.summary.image(name='input_images', tensor=inputs)
    tf.summary.histogram(name='labels', values=labels)
    # learning rate
    self.learning_rate = tf.Variable(learning_rate, name='learning_rate',
      trainable=False)
    self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * decay)
    tf.summary.scalar(name='learning_rate', tensor=self.learning_rate)
    # model
    with tf.name_scope('conv1'):
      h1_size = 32
      w = tf.get_variable(name='conv_w1',
        shape=[7, 7, input_channel, h1_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.025))
      b = tf.get_variable(name='conv_b1', shape=[h1_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv2'):
      h2_size = 32
      w = tf.get_variable(name='conv_w2',
        shape=[5, 5, h1_size, h2_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.05))
      b = tf.get_variable(name='conv_b2', shape=[h2_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv3'):
      h3_size = 64
      w = tf.get_variable(name='conv_w3',
        shape=[5, 5, h2_size, h3_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.03))
      b = tf.get_variable(name='conv_b3', shape=[h3_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv4'):
      h4_size = 64
      w = tf.get_variable(name='conv_w4',
        shape=[5, 5, h3_size, h4_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.03))
      b = tf.get_variable(name='conv_b4', shape=[h4_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv5'):
      h5_size = 64
      w = tf.get_variable(name='conv_w5',
        shape=[5, 5, h4_size, h5_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.03))
      b = tf.get_variable(name='conv_b5', shape=[h5_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv6'):
      h6_size = 64
      w = tf.get_variable(name='conv_w6',
        shape=[5, 5, h5_size, h6_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.03))
      b = tf.get_variable(name='conv_b6', shape=[h4_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('fc7'):
      h_shape = h.get_shape().as_list()
      connect_size = h_shape[1] * h_shape[2] * h_shape[3]
      h7_size = 1024
      w = tf.get_variable(name='w7',
        shape=[connect_size, h7_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / connect_size)))
      b = tf.get_variable(name='b7', shape=[h7_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.matmul(tf.reshape(h, [-1, connect_size]), w) + b)
    with tf.name_scope('fc8'):
      h8_size = 256
      w = tf.get_variable(name='w8',
        shape=[h7_size, h8_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / connect_size)))
      b = tf.get_variable(name='b8', shape=[h8_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.matmul(h, w) + b)
    with tf.name_scope('output'):
      w = tf.get_variable(name='ow', shape=[h8_size, output_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / h5_size)))
      b = tf.get_variable(name='ob', shape=[output_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      logits = tf.matmul(h, w) + b
      outputs = tf.nn.softmax(logits)
      tf.summary.histogram(name='outputs', values=outputs)
    # loss and optimizer
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
      tf.summary.scalar(name='loss', tensor=loss)
      accuracy = tf.reduce_sum(tf.cast(tf.equal(
        tf.argmax(outputs, axis=1), tf.argmax(labels, axis=1)), tf.float32)) * \
        100.0 / tf.cast(tf.shape(labels)[0], tf.float32)
      tf.summary.scalar(name='accuracy', tensor=accuracy)
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      train_op = optimizer.minimize(loss)
    return inputs, labels, keep_prob, outputs, loss, accuracy, train_op

  def load(self, sess, checkpoint_path):
    if os.path.isfile(checkpoint_path + '.meta') and \
        os.path.isfile(checkpoint_path + '.index'):
      self.logger.info('loading last %s checkpoint...' % (checkpoint_path))
      self.saver.restore(sess, checkpoint_path)
      self.start_epoch = int(checkpoint_path.split('-')[-1].strip())
    else:
      self.logger.warning('%s does not exists' % (checkpoint_path))

  def train(self, sess, data, label,
      batch_size=1024, output_period=1000,
      keep_prob=0.8, max_epoch=10000,
      decay_epoch=1000):
    # initialize
    if self.start_epoch == 0:
      self.logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())
    # training
    self.logger.info('start training...')
    last = time.time()
    for epoch in range(self.start_epoch, self.start_epoch + max_epoch + 1):
      # prepare batch data
      offset = (epoch * batch_size) % (data.shape[0] - batch_size + 1)
      batch_data = data[offset:offset+batch_size, :]
      batch_label = label[offset:offset+batch_size, :]
      _, loss = sess.run([self.train_op, self.loss], feed_dict={
        self.inputs: batch_data,
        self.labels: batch_label,
        self.keep_prob: keep_prob
      })
      # output
      if epoch % output_period == 0:
        feed_dict={
          self.inputs: batch_data,
          self.labels: batch_label,
          self.keep_prob: 1.0
        }
        ms, loss, accuracy = sess.run(
          [self.merged_summary, self.loss, self.accuracy], feed_dict)
        self.logger.info('%d. loss: %f | accuracy: %f | time used: %f' %
          (epoch, loss, accuracy, (time.time() - last)))
        last = time.time()
        if self.saving:
          self.saver.save(sess, self.checkpoint_path, global_step=epoch)
          self.summary_writer.add_summary(ms, global_step=epoch)
      if epoch % decay_epoch == 0 and epoch != 0:
        sess.run(self.decay_lr)

  def predict(self, sess, data):
    return sess.run(self.outputs, feed_dict={
      self.inputs: data,
      self.keep_prob: 1.0
    })


def test():
  image_size = 28
  image_channel = 1
  data = np.random.randn(256, image_size, image_size, image_channel)
  label = np.zeros(shape=[256, 10])
  label[:, 0] = 1.0
  model = MNISTConvolutionModel(image_size, image_size, image_channel,
    10, model_name='test', saving=False)

  with tf.Session() as sess:
    model.train(sess, data, label,
      batch_size=32,
      output_period=10,
      keep_prob=0.8,
      max_epoch=100)


if __name__ == '__main__':
  test()

import tensorflow as tf
import numpy as np
import os
import time
import logging


class KinectPoseModel(object):
  def __init__(self, input_width, input_height, input_channel, output_size,
      model_name='KinectPoseModel', learning_rate=1e-3, decay=0.9, saving=True):
    # logger
    logging.basicConfig()
    self.logger = logging.getLogger(model_name)
    self.logger.setLevel(logging.INFO)
    self.logger.info('setting up model...')
    # model
    with tf.variable_scope(model_name):
      self.prev_images, self.next_images, self.labels, self.keep_prob, \
        self.outputs, self.loss, self.error, self.train_op = \
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
    prev_images = tf.placeholder(dtype=tf.float32,
      shape=[None, input_height, input_width, input_channel])
    tf.summary.image(name='prev_images', tensor=prev_images)
    next_images = tf.placeholder(dtype=tf.float32,
      shape=[None, input_height, input_width, input_channel])
    tf.summary.image(name='next_images', tensor=next_images)
    images = tf.concat([prev_images, next_images], axis=3)
    images = tf.image.resize_area(images, [input_height / 8, input_width / 8])
    # labels
    labels = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
    tf.summary.histogram(name='labels', values=labels)
    keep_prob = tf.placeholder(dtype=tf.float32, shape=())
    # learning rate
    self.learning_rate = tf.Variable(learning_rate, name='learning_rate',
      trainable=False)
    self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * decay)
    tf.summary.scalar(name='learning_rate', tensor=self.learning_rate)
    # model
    with tf.name_scope('conv1'):
      h1_size = 32
      w = tf.get_variable(name='conv_w1',
        shape=[3, 3, input_channel * 2, h1_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b1', shape=[h1_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(images, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv2'):
      h2_size = 32
      w = tf.get_variable(name='conv_w2',
        shape=[3, 3, h1_size, h2_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b2', shape=[h2_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv3'):
      h3_size = 32
      w = tf.get_variable(name='conv_w3',
        shape=[3, 3, h2_size, h3_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b3', shape=[h3_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv4'):
      h4_size = 32
      w = tf.get_variable(name='conv_w4',
        shape=[3, 3, h3_size, h4_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b4', shape=[h4_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv5'):
      h5_size = 32
      w = tf.get_variable(name='conv_w5',
        shape=[3, 3, h4_size, h5_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b5', shape=[h5_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv6'):
      h6_size = 32
      w = tf.get_variable(name='conv_w6',
        shape=[3, 3, h5_size, h6_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b6', shape=[h6_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('fc7'):
      h_size = h.get_shape().as_list()
      self.logger.info('connect size: %s' % (str(h_size)))
      connect_size = h_size[1] * h_size[2] * h_size[3]
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
          stddev=np.sqrt(2.0 / h7_size)))
      b = tf.get_variable(name='b8', shape=[h8_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.matmul(h, w) + b)
    with tf.name_scope('output'):
      w = tf.get_variable(name='ow', shape=[h8_size, output_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(10.0 / h8_size)))
      b = tf.get_variable(name='ob', shape=[output_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      logits = tf.matmul(h, w) + b
      outputs = logits
      tf.summary.histogram(name='outputs', values=outputs)
    # loss and optimizer
    with tf.name_scope('loss'):
      # scale up to cm as unit
      loss = tf.reduce_mean(tf.square(logits - labels)) * output_size * (10 **2)
      tf.summary.scalar(name='loss', tensor=loss)
      error = tf.reduce_mean(tf.abs(logits - labels))
      tf.summary.scalar(name='error', tensor=error)
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      train_op = optimizer.minimize(loss)
    return prev_images, next_images, labels, keep_prob, outputs, \
        loss, error, train_op

  def load(self, sess, checkpoint_path):
    if os.path.isfile(checkpoint_path + '.meta') and \
        os.path.isfile(checkpoint_path + '.index'):
      self.logger.info('loading last %s checkpoint...' % (checkpoint_path))
      self.saver.restore(sess, checkpoint_path)
      self.start_epoch = int(checkpoint_path.split('-')[-1].strip())
    else:
      self.logger.warning('%s does not exists' % (checkpoint_path))

  def train(self, sess, prev_images, next_images, label,
      batch_size=1024, output_period=10,
      keep_prob=0.8, max_epoch=100000):
    # initialize
    if self.start_epoch == 0:
      self.logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())
    # training
    self.logger.info('start training...')
    last = time.time()
    for epoch in range(self.start_epoch, self.start_epoch + max_epoch + 1):
      # prepare batch data
      offset = (epoch * batch_size) % (prev_images.shape[0] - batch_size + 1)
      batch_prev = prev_images[offset:offset+batch_size, :]
      batch_next = next_images[offset:offset+batch_size, :]
      batch_label = label[offset:offset+batch_size, :]
      _, loss = sess.run([self.train_op, self.loss], feed_dict={
        self.prev_images: batch_prev,
        self.next_images: batch_next,
        self.labels: batch_label,
        self.keep_prob: keep_prob
      })
      # output
      if epoch % output_period == 0:
        feed_dict={
          self.prev_images: batch_prev,
          self.next_images: batch_next,
          self.labels: batch_label,
          self.keep_prob: 1.0
        }
        ms, loss, error = sess.run(
          [self.merged_summary, self.loss, self.error], feed_dict)
        self.logger.info('%d. loss: %f | error: %f | time used: %f' %
          (epoch, loss, error, (time.time() - last)))
        last = time.time()
        if self.saving:
          self.saver.save(sess, self.checkpoint_path, global_step=epoch)
          self.summary_writer.add_summary(ms, global_step=epoch)

  def train_with_loader(self, sess, data_loader,
      batch_size=1024, output_period=10, decay_epoch=1000,
      keep_prob=0.8, max_epoch=100000):
    # initialize
    if self.start_epoch == 0:
      self.logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())
    # training
    self.logger.info('start training...')
    last = time.time()
    for epoch in range(self.start_epoch, self.start_epoch + max_epoch + 1):
      # prepare batch data
      p, n, l = data_loader.diff_depth_batch(batch_size)
      _, loss = sess.run([self.train_op, self.loss], feed_dict={
        self.prev_images: p,
        self.next_images: n,
        self.labels: l,
        self.keep_prob: keep_prob
      })
      # output
      if epoch % output_period == 0:
        feed_dict={
          self.prev_images: p,
          self.next_images: n,
          self.labels: l,
          self.keep_prob: 1.0
        }
        ms, loss, error = sess.run(
          [self.merged_summary, self.loss, self.error], feed_dict)
        self.logger.info('%d. loss: %f | error: %f | time used: %f' %
          (epoch, loss, error, (time.time() - last)))
        last = time.time()
        if self.saving:
          self.saver.save(sess, self.checkpoint_path, global_step=epoch)
          self.summary_writer.add_summary(ms, global_step=epoch)
      if epoch % decay_epoch == 0 and epoch != 0:
        self.logger.info('decay learning rate...')
        sess.run(self.decay_lr)

  def predict(self, sess, prev_images, next_images):
    return sess.run(self.outputs, feed_dict={
      self.prev_images: prev_images,
      self.next_images: next_images,
      self.keep_prob: 1.0
    })


def test():
  input_width = 640
  input_height = 480
  input_channel = 3
  output_size = 7

  # test data
  test_batch_size = 256
  data1 = np.random.randn(test_batch_size,
    input_height, input_width, input_channel)
  data2 = np.random.randn(test_batch_size,
    input_height, input_width, input_channel)
  label = np.zeros(shape=[test_batch_size, output_size])
  label[:, 0] = 1.0

  model = KinectPoseModel(input_width, input_height, input_channel,
    output_size, model_name='test', saving=False)

  with tf.Session() as sess:
    model.train(sess, data1, data2, label,
      batch_size=32,
      output_period=10,
      keep_prob=0.8,
      max_epoch=100)


if __name__ == '__main__':
  test()

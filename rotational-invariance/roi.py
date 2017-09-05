import tensorflow as tf
import numpy as np
import logging
import os
import time


class ROIModel(object):
  def __init__(self, input_width, input_height, input_channel,
      output_width, output_height,
      model_name='ROIModel',
      learning_rate=1e-3, decay=0.9, saving=True):
    # logger
    logging.basicConfig()
    self.logger = logging.getLogger(model_name)
    self.logger.setLevel(logging.INFO)
    self.logger.info('setting up model...')
    # model
    with tf.variable_scope(model_name):
      self.inputs, self.indicator_label, self.xy_label, self.size_label, \
        self.keep_prob, self.outputs, self.loss, \
        self.accuracy, self.train_op = \
        self._build_model(input_width, input_height, input_channel,
          output_width, output_height,
          learning_rate, decay)
    # checkpoint
    self.start_epoch = 0
    self.saving = saving
    self.saver = tf.train.Saver()
    if saving:
      self.checkpoint_path, summary_path = self._prepare_save_dir(model_name)
      # saver
      self.logger.info('setting up saver...')
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
      output_width, output_height,
      learning_rate, decay,
      lambda_indicator=1.0, lambda_coord=3.0, lambda_noobj=0.3):
    with tf.device('/cpu:0'):
      # inputs
      inputs = tf.placeholder(dtype=tf.float32,
        shape=[None, input_height, input_width, input_channel],
        name='input_images')
      indicator_label = tf.placeholder(dtype=tf.float32,
        shape=[None, output_height, output_width, 1],
        name='indicator_label')
      xy_label = tf.placeholder(dtype=tf.float32,
        shape=[None, output_height, output_width, 2],
        name='xy_label')
      size_label = tf.placeholder(dtype=tf.float32,
        shape=[None, output_height, output_width, 2],
        name='size_label')
      keep_prob = tf.placeholder(dtype=tf.float32, shape=[],
        name='keep_prob')
    tf.summary.image(name='input_images', tensor=inputs)
    tf.summary.image(name='indicator_label', tensor=indicator_label)
    tf.summary.histogram(name='xy_label', values=xy_label)
    tf.summary.histogram(name='size_label', values=size_label)
    # learning rate
    self.learning_rate = tf.Variable(learning_rate, name='learning_rate',
      trainable=False)
    self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * decay)
    tf.summary.scalar(name='learning_rate', tensor=self.learning_rate)
    # model
    with tf.name_scope('conv1'):
      h1_size = 32
      w = tf.get_variable(name='conv_w1',
        shape=[5, 5, input_channel, h1_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.06))
      b = tf.get_variable(name='conv_b1', shape=[h1_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(inputs / 60.0, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    tf.summary.histogram('conv1_output', values=h)
    with tf.name_scope('conv2'):
      h2_size = 64
      w = tf.get_variable(name='conv_w2',
        shape=[5, 5, h1_size, h2_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.036))
      b = tf.get_variable(name='conv_b2', shape=[h2_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob)
    tf.summary.histogram('conv2_output', values=h)
    with tf.name_scope('conv3'):
      h3_size = 256
      w = tf.get_variable(name='conv_w3',
        shape=[5, 5, h2_size, h3_size],
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
    tf.summary.histogram('conv3_output', values=h)
    with tf.name_scope('fc4'):
      h_shape = h.get_shape().as_list()
      connect_size = h_shape[1] * h_shape[2] * h_shape[3]
      h4_size = 2048
      w = tf.get_variable(name='w4',
        shape=[connect_size, h4_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / connect_size)))
      b = tf.get_variable(name='b4', shape=[h4_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.matmul(tf.reshape(h, [-1, connect_size]), w) + b)
    tf.summary.histogram('fc4_output', values=h)
    fully_connect_size = h4_size
    with tf.name_scope('output'):
      with tf.name_scope('indicator'):
        w = tf.get_variable(name='indicator_ow',
          shape=[fully_connect_size, output_height * output_width],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / fully_connect_size)))
        b = tf.get_variable(name='indicator_ob',
          shape=[output_height * output_width],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
        indicator_logits = tf.reshape(tf.matmul(h, w) + b,
          [-1, output_height, output_width, 1])
      tf.summary.image(name='indicator_output', tensor=indicator_logits)
      with tf.name_scope('xy'):
        w = tf.get_variable(name='xy_ow',
          shape=[fully_connect_size, output_height * output_width * 2],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / fully_connect_size)))
        b = tf.get_variable(name='xy_ob',
          shape=[output_height * output_width * 2],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
        xy_logits = tf.reshape(tf.matmul(h, w) + b,
          [-1, output_height, output_width, 2])
      tf.summary.histogram(name='xy_output', values=xy_logits)
      with tf.name_scope('size'):
        w = tf.get_variable(name='size_ow',
          shape=[fully_connect_size, output_height * output_width * 2],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / fully_connect_size)))
        b = tf.get_variable(name='size_ob',
          shape=[output_height * output_width * 2],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
        size_logits = tf.reshape(tf.matmul(h, w) + b,
          [-1, output_height, output_width, 2])
      tf.summary.histogram(name='size_output', values=size_logits)
      outputs = tf.concat([indicator_logits, xy_logits, size_logits],
        axis=3)
    # loss and optimizer
    with tf.name_scope('loss'):
      valid = tf.cast(tf.equal(indicator_label, 1), tf.float32)
      roi_count = tf.reduce_sum(valid)
      invalid = 1.0 - valid
      with tf.name_scope('indicator'):
        indicator_loss = tf.reduce_sum(valid *
          tf.square(indicator_label - indicator_logits)) / roi_count
        noobj_loss = tf.reduce_sum(invalid *
          tf.square(indicator_label - indicator_logits)) / roi_count
      accuracy = tf.reduce_sum(tf.cast(tf.equal(
        tf.greater(indicator_logits, 0.5),
        tf.greater(indicator_label, 0.5)), tf.float32)) * 100.0 / \
          output_width / output_height / \
          tf.cast(tf.shape(inputs)[0], tf.float32)
      tf.summary.scalar(name='indicator_loss', tensor=indicator_loss)
      tf.summary.scalar(name='noobj_loss', tensor=noobj_loss)
      tf.summary.scalar(name='indicator_accuracy', tensor=accuracy)
      with tf.name_scope('xy'):
        xy_loss = tf.reduce_sum(valid *
          tf.square(xy_logits - xy_label)) / roi_count
        xy_error = tf.reduce_sum(valid *
          tf.abs(xy_logits - xy_label)) / roi_count
      tf.summary.scalar(name='xy_loss', tensor=xy_loss)
      tf.summary.scalar(name='xy_error', tensor=xy_error)
      with tf.name_scope('size'):
        size_loss = tf.reduce_sum(valid *
          tf.square(size_logits - size_label)) / roi_count
        size_error = tf.reduce_sum(valid *
          tf.abs(size_logits - size_label)) / roi_count
      tf.summary.scalar(name='size_loss', tensor=size_loss)
      tf.summary.scalar(name='size_error', tensor=size_error)
      total_loss = lambda_coord * xy_loss + \
        lambda_coord * size_loss + \
        lambda_indicator * indicator_loss + \
        lambda_noobj * noobj_loss
      tf.summary.scalar(name='total_loss', tensor=total_loss)
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      train_op = optimizer.minimize(total_loss)
    return inputs, indicator_label, xy_label, size_label, keep_prob, \
      outputs, total_loss, accuracy, train_op

  def load(self, sess, checkpoint_path):
    if os.path.isfile(checkpoint_path + '.meta') and \
        os.path.isfile(checkpoint_path + '.index'):
      self.logger.info('loading last %s checkpoint...' % (checkpoint_path))
      self.saver.restore(sess, checkpoint_path)
      self.start_epoch = int(checkpoint_path.split('-')[-1].strip())
    else:
      self.logger.warning('%s does not exists' % (checkpoint_path))

  def train_batch(self, sess, batch_data,
      batch_indicator_label, batch_xy_label, batch_size_label,
      keep_prob):
    _, loss = sess.run([self.train_op, self.loss], feed_dict={
      self.inputs: batch_data,
      self.indicator_label: batch_indicator_label,
      self.xy_label: batch_xy_label,
      self.size_label: batch_size_label,
      self.keep_prob: keep_prob
    })
    return loss

  def train(self, sess, input_image,
      indicator_label, xy_label, size_label,
      batch_size=256, output_period=1000,
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
      indexes = np.random.permutation(
        np.arange(len(input_image)))[:batch_size]
      batch_data = input_image[indexes, :]
      batch_indicator_label = indicator_label[indexes, :]
      batch_xy_label = xy_label[indexes, :]
      batch_size_label = size_label[indexes, :]
      loss = self.train_batch(sess, batch_data,
        batch_indicator_label, batch_xy_label, batch_size_label, keep_prob)
      # output
      if epoch % output_period == 0:
        feed_dict = {
          self.inputs: batch_data,
          self.indicator_label: batch_indicator_label,
          self.xy_label: batch_xy_label,
          self.size_label: batch_size_label,
          self.keep_prob: 1.0
        }
        ms, loss, accuracy = sess.run(
          [self.merged_summary, self.loss, self.accuracy], feed_dict)
        self.logger.info('%d. loss: %f | accuracy: %f | time used: %f' %
                (epoch, loss, accuracy, (time.time() - last)))
        last = time.time()
        # save
        if self.saving:
          self.saver.save(sess, self.checkpoint_path, global_step=epoch)
          self.summary_writer.add_summary(ms, global_step=epoch)
      # learning rate decay
      if epoch % decay_epoch == 0 and epoch != 0:
        sess.run(self.decay_lr)

  def predict(self, sess, data):
    return sess.run(self.outputs, feed_dict={
      self.inputs: data,
      self.keep_prob: 1.0
    })


def test():
  input_width, input_height, input_channel = 200, 150, 3
  output_width, output_height = 16, 12

  # prepare fake data
  test_batch_size = 1024
  data = np.random.randn(test_batch_size,
    input_height, input_width, input_channel) * 128
  data = data.astype(np.float32)
  label = np.random.randn(test_batch_size,
    output_height, output_width, 5)
  label = label.astype(np.float32)

  model = ROIModel(input_width, input_height, input_channel,
    output_width, output_height,
    model_name='test', saving=False)

  with tf.Session() as sess:
    model.train(sess, data,
      label[:, :, :, 0:1],
      label[:, :, :, 1:3],
      label[:, :, :, 3:5],
      batch_size=256,
      output_period=10,
      keep_prob=0.8,
      max_epoch=100)


if __name__ == '__main__':
  test()

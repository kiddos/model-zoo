from __future__ import print_function
import tensorflow as tf
import numpy as np
import logging
import os
import time
import math


class RotationalInvarianceModel(object):
  def __init__(self, image_size, channel_size, output_size,
      model_name='RotationModel', learning_rate=1e-3):
    logging.basicConfig()
    self.logger = logging.getLogger(model_name)
    self.logger.setLevel(logging.INFO)
    self.logger.info('setting up model...')
    with tf.variable_scope(model_name):
      self.inputs, self.labels, self.rotations, self.keep_prob, \
          self.outputs, self.rotation_predictions, \
          self.loss, self.accuracy, self.rotation_error, \
          self.train_op = self._build_model(image_size, channel_size,
              output_size, learning_rate)
    self.start_epoch = 0
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
    self.logger.info('creating model&summary directory...')
    model_path = model_name + str(index)
    os.mkdir(model_path)
    summary_path = os.path.join(model_path, 'summary')
    os.mkdir(summary_path)
    checkpoint_path = os.path.join(model_path, model_name)
    return checkpoint_path, summary_path

  def _build_model(self, image_size, channel_size, output_size, lr, decay=0.9):
    # learning rate
    self.learning_rate = tf.Variable(lr, name='learning_rate', trainable=False)
    self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * decay)
    tf.summary.scalar(name='learning_rate',
        tensor=self.learning_rate)

    inputs = tf.placeholder(dtype=tf.float32,
        shape=[None, image_size, image_size, 3])
    labels = tf.placeholder(dtype=tf.float32,
        shape=[None, output_size])
    rotations = tf.placeholder(dtype=tf.float32,
        shape=[None, 1])
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

    tf.summary.image(name='inputs_image', tensor=inputs, max_outputs=3)
    tf.summary.histogram(name='inputs_label', values=labels)
    tf.summary.histogram(name='rotations', values=rotations)

    # model
    with tf.name_scope('convolution_1'):
      h1_size = 64
      w = tf.get_variable(name='cw1', shape=[5, 5, 3, h1_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.01))
      tf.summary.image(name='convolution_1_filter',
          tensor=tf.reshape(w, [-1, 5, 5, 4]), max_outputs=3)
      b = tf.get_variable(name='cb1', shape=[h1_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
      tf.summary.image(name='convolution_1_output',
          tensor=tf.reshape(h, [-1, image_size, image_size, 1]),
          max_outputs=3)
      with tf.name_scope('max_pool_1'):
        h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1],
            padding='SAME')
      with tf.name_scope('dropout_1'):
        h = tf.nn.dropout(h, keep_prob)

    with tf.name_scope('convolution_2'):
      h2_size = 64
      w = tf.get_variable(name='cw2', shape=[5, 5, h1_size, h2_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.01))
      tf.summary.image(name='convolution_2_filter',
          tensor=tf.reshape(w, [-1, 5, 5, 4]), max_outputs=3)
      b = tf.get_variable(name='cb2', shape=[h2_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
      tf.summary.image(name='convolution_2_output',
          tensor=tf.reshape(h, [-1, image_size / 2, image_size / 2, 1]),
          max_outputs=3)
      with tf.name_scope('max_pool_2'):
        h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1],
            padding='SAME')
      with tf.name_scope('dropout_2'):
        h = tf.nn.dropout(h, keep_prob)

    with tf.name_scope('convolution_3'):
      h3_size = 64
      w = tf.get_variable(name='cw3', shape=[5, 5, h2_size, h3_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.01))
      tf.summary.image(name='convolution_3_filter',
          tensor=tf.reshape(w, [-1, 5, 5, 1]), max_outputs=3)
      b = tf.get_variable(name='cb3', shape=[h3_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
      tf.summary.image(name='convolution_3_output',
          tensor=tf.reshape(h, [-1, image_size / 4, image_size / 4, 1]),
          max_outputs=3)
      with tf.name_scope('max_pool_3'):
        h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1],
            padding='SAME')
      with tf.name_scope('dropout_3'):
        h = tf.nn.dropout(h, keep_prob)

    with tf.name_scope('fully_connected_layer_4'):
      connect_size = image_size * image_size / 4 / 4 / 4 * h3_size
      h4_size = 1024
      w = tf.get_variable(name='fw4', shape=[connect_size, h4_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.001))
      b = tf.get_variable(name='fb4', shape=[h4_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      fh = tf.nn.relu(tf.matmul(tf.reshape(h, [-1, connect_size]), w) + b)

    with tf.name_scope('classes'):
      with tf.name_scope('fully_connected_layer_5'):
        ch5_size = 1024
        w = tf.get_variable(name='cw5', shape=[h4_size, ch5_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/h4_size)))
        b = tf.get_variable(name='cb5', shape=[ch5_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        ch = tf.nn.relu(tf.matmul(fh, w) + b)

      with tf.name_scope('fully_connected_output'):
        w = tf.get_variable(name='cow', shape=[ch5_size, output_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/ch5_size)))
        b = tf.get_variable(name='cob', shape=[output_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        logits = tf.matmul(ch, w) + b
        outputs = tf.nn.softmax(logits)
        tf.summary.histogram(name='class_outputs', values=outputs)

    with tf.name_scope('rotation'):
      with tf.name_scope('fully_connected_layer_4'):
        rh5_size = 1024
        w = tf.get_variable(name='rw4', shape=[h4_size, rh5_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/h4_size)))
        b = tf.get_variable(name='rb4', shape=[rh5_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
        rh = tf.nn.relu(tf.matmul(fh, w) + b)

      with tf.name_scope('fully_connected_layer_5'):
        rh6_size = 1024
        w = tf.get_variable(name='rw5', shape=[rh5_size, rh6_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/rh5_size)))
        b = tf.get_variable(name='rb5', shape=[rh6_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        rh = tf.nn.relu(tf.matmul(rh, w) + b)

      with tf.name_scope('fully_connected_layer_6'):
        rh7_size = 1024
        w = tf.get_variable(name='rw6', shape=[rh6_size, rh7_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/rh6_size)))
        b = tf.get_variable(name='rb6', shape=[rh7_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        rh = tf.nn.relu(tf.matmul(rh, w) + b)

      with tf.name_scope('fully_connected_output'):
        w = tf.get_variable(name='row', shape=[rh7_size, 1],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/rh7_size)))
        b = tf.get_variable(name='rob', shape=[1],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        rotation_predictions = tf.atan(tf.matmul(rh, w) + b) * 2.0
        tf.summary.histogram(name='rotation_predictions',
            values=rotation_predictions)

    # loss and optimizer
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)) + \
          tf.reduce_mean(tf.pow(rotation_predictions - rotations, 2))
      tf.summary.scalar(name='loss', tensor=loss)
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      train_op = optimizer.minimize(loss)
    with tf.name_scope('accuracy'):
      prediction = tf.argmax(outputs, axis=1)
      answers = tf.argmax(labels, axis=1)
      print(tf.shape(prediction)[0])
      accuracy = tf.reduce_sum(tf.cast(
          tf.equal(prediction, answers), tf.float32)) * \
          100.0 / tf.cast(tf.shape(prediction)[0], tf.float32)
      tf.summary.scalar(name='accuracy', tensor=accuracy)
      rotation_error = tf.reduce_mean(tf.abs(rotation_predictions - rotations))
      tf.summary.scalar(name='rotation_error', tensor=rotation_error)
    return inputs, labels, rotations, keep_prob, \
        outputs, rotation_predictions, \
        loss, accuracy, rotation_error, train_op

  def load(self, sess, checkpoint_path):
    if os.path.isfile(checkpoint_path + '.meta') and \
      os.path.isfile(checkpoint_path + '.index'):
      self.logger.info('loading last %s checkpoint...' % (checkpoint_path))
      self.saver.restore(sess, checkpoint_path)
      self.start_epoch = int(checkpoint_path.split('-')[-1].strip())
    else:
      self.logger.warning('%s does not exists' % (checkpoint_path))

  def train(self, sess, data, label, rotations,
      batch_size=1024, output_period=1000, decay_epoch=1000,
      max_epoch=10000, keep_prob=0.8, variance=1.0):
    # initialize
    if self.start_epoch == 0:
      self.logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())
    # training
    self.logger.info('start training...')
    last = time.time()
    for epoch in range(self.start_epoch, self.start_epoch + max_epoch + 1):
      offset = (epoch * batch_size) % (data.shape[0] - batch_size + 1)
      batch_data = data[offset:offset+batch_size, :]
      batch_label = label[offset:offset+batch_size, :]
      batch_rotation = rotations[offset:offset+batch_size, :]
      # add variance
      batch_data = batch_data.astype(np.float32) + \
          variance * np.random.randn(batch_size,
          data.shape[1], data.shape[2], data.shape[3])
      # train op
      _, loss = sess.run([self.train_op, self.loss], feed_dict={
        self.inputs: batch_data,
        self.labels: batch_label,
        self.rotations: batch_rotation,
        self.keep_prob: keep_prob
      })
      # display
      if epoch % output_period == 0:
        feed_dict={
          self.inputs: batch_data,
          self.labels: batch_label,
          self.rotations: batch_rotation,
          self.keep_prob: 1.0
        }
        ms, loss, accuracy, rotation_error = sess.run([self.merged_summary,
            self.loss, self.accuracy, self.rotation_error], feed_dict)
        self.summary_writer.add_summary(ms, global_step=epoch)

        self.logger.info(('%d. loss: %f | accuracy: %f | error: %s ' +
            ' 1 batch time used: %f (sec) | saving...') %
            (epoch, loss, accuracy, rotation_error,
                (time.time() - last) / output_period))
        last = time.time()
        self.saver.save(sess, self.checkpoint_path, global_step=epoch)
      if epoch % decay_epoch == 0 and epoch > 0:
        self.logger.info('decay leanring rate...')
        sess.run(self.decay_lr)

  def predict(self, sess, data):
    return sess.run([self.outputs, self.rotation_predictions], feed_dict={
      self.inputs: data,
      self.keep_prob: 1.0
    })


def test():
  test_data = np.random.randn(1, 64, 64, 3) * 127 + 128
  test_label = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
  test_rotation = np.random.randn(1, 1) * 3.14
  model = RotationalInvarianceModel(
      image_size=64, channel_size=3, output_size=10)

  with tf.Session() as sess:
    model.train(sess, test_data, test_label, test_rotation,
        batch_size=1, output_period=3, max_epoch=30)


if __name__ == '__main__':
  test()

from __future__ import print_function
import numpy as np
import tensorflow as tf
import logging
import os
import time


class YOLOModel(object):
  def __init__(self, image_width, image_height, channel_size,
      output_width, output_height, output_classes, bounding_box=1,
      model_name='YOLO', learning_rate=1e-3, decay=0.9):
    logging.basicConfig()
    self.logger = logging.getLogger(model_name)
    self.logger.setLevel(logging.INFO)
    self.logger.info('setting up model...')
    with tf.variable_scope(model_name):
      self.inputs, self.target_coordinate, self.target_dimension, \
        self.target_class, self.keep_prob, self.coordinate, self.dimension, \
        self.predict_class, self.loss, self.train_op = \
        self._build_model(image_width, image_height, channel_size,
          output_width, output_height, output_classes, bounding_box,
          learning_rate, decay)
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

  def _build_model(self, image_width, image_height, channel_size,
      output_width, output_height, bounding_box, output_classes, lr, decay=0.9,
      lambda_coord=1.0, lambda_no_obj=0.1):
    inputs = tf.placeholder(dtype=tf.float32,
      shape=[None, image_height, image_width, channel_size],
      name='input_images')
    target_coordinate = tf.placeholder(dtype=tf.float32,
      shape=[None, output_height, output_width,
        bounding_box * 2 * output_classes],
      name='target_coordinate')
    target_dimension = tf.placeholder(dtype=tf.float32,
      shape=[None, output_height, output_width,
        bounding_box * 2 * output_classes],
      name='target_dimension')
    target_class = tf.placeholder(dtype=tf.float32,
      shape=[None, output_height, output_width,
        bounding_box * output_classes],
      name='target_class')
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    tf.summary.image(name='input_image', tensor=inputs)
    tf.summary.histogram(name='target_coordinate',
      values=target_coordinate)
    tf.summary.histogram(name='target_dimension',
      values=target_dimension)
    tf.summary.image(name='target_class',
      tensor=target_class)

    self.learning_rate = tf.Variable(lr, name='learning_rate', dtype=tf.float32)
    self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * decay)
    tf.summary.scalar(name='learning rate', tensor=self.learning_rate)
    # model
    with tf.name_scope('convolution_1'):
      h1_size = 32
      w = tf.get_variable(name='conv_w1', shape=[5, 5, channel_size, h1_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b1', shape=[h1_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
      tf.summary.image(name='conv1_output',
        tensor=tf.reshape(h, [-1, image_height, image_width, 1]))
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
          padding='SAME')
      h = tf.nn.dropout(h, keep_prob=keep_prob)
    with tf.name_scope('convolution_2'):
      h2_size = 64
      w = tf.get_variable(name='conv_w2', shape=[3, 3, h1_size, h2_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b2', shape=[h2_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      tf.summary.image(name='conv2_output',
        tensor=tf.reshape(h, [-1, image_height / 2, image_width / 2, 1]))
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob=keep_prob)
    with tf.name_scope('convolution_3'):
      h3_size = 64
      w = tf.get_variable(name='conv_w3', shape=[3, 3, h2_size, h3_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b3', shape=[h3_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      tf.summary.image(name='conv3_output',
        tensor=tf.reshape(h, [-1, image_height / 4, image_width / 4, 1]))
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob=keep_prob)
    with tf.name_scope('convolution_4'):
      h4_size = 64
      w = tf.get_variable(name='conv_w4', shape=[3, 3, h3_size, h4_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable(name='conv_b4', shape=[h4_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      tf.summary.image(name='conv4_output',
        tensor=tf.reshape(h, [-1, image_height / 8, image_width / 8, 1]))
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.dropout(h, keep_prob=keep_prob)
    with tf.name_scope('fully_connected_5'):
      #  connect_size = image_width * image_height / 4 / 4 / 4 / 4 * h4_size
      #  connect_size = h.get_shape()[1] * h.get_shape()[2] * h.get_shape()[3]
      connect_size = 8 * 10 * h4_size
      h5_size = 2048
      w = tf.get_variable(name='fw5', shape=[connect_size, h5_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / connect_size)))
      b = tf.get_variable(name='fb5', shape=[h5_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      fh = tf.nn.relu(tf.matmul(tf.reshape(h, [-1, connect_size]), w) + b)
    with tf.name_scope('fully_connected_6'):
      h6_size = 1024
      w = tf.get_variable(name='fw6', shape=[h5_size, h6_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / h5_size)))
      b = tf.get_variable(name='fb6', shape=[h6_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      fh = tf.nn.relu(tf.matmul(fh, w) + b)
    with tf.name_scope('fully_connect_output'):
      with tf.name_scope('coordinate'):
        output_size = output_width * output_height * 2 * output_classes
        w = tf.get_variable(name='coord_ow', shape=[h6_size, output_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / h6_size)))
        b = tf.get_variable(name='coord_ob', shape=[output_size],
          dtype=tf.float32,
          initializer=tf.zeros_initializer())
        coordinate = tf.reshape(tf.matmul(fh, w) + b,
          [-1, output_height, output_width, 2])
        tf.summary.histogram(name='coordinate', values=coordinate)
      with tf.name_scope('dimension'):
        output_size = output_width * output_height * 2 * output_classes
        w = tf.get_variable(name='dim_ow', shape=[h6_size, output_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / h6_size)))
        b = tf.get_variable(name='dim_ob', shape=[output_size],
          dtype=tf.float32,
          initializer=tf.zeros_initializer())
        dimension = tf.reshape(tf.matmul(fh, w) + b,
          [-1, output_height, output_width, 2])
        tf.summary.histogram(name='dimension', values=dimension)
      with tf.name_scope('target_class'):
        output_size = output_width * output_height * output_classes
        w = tf.get_variable(name='class_ow', shape=[h6_size, output_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / h6_size)))
        b = tf.get_variable(name='class_ob', shape=[output_size],
          dtype=tf.float32,
          initializer=tf.zeros_initializer())
        predict_class = tf.reshape(tf.matmul(fh, w) + b,
          [-1, output_height, output_width, 1])
        tf.summary.image(name='predict_class',
          tensor=tf.reshape(predict_class,
            [-1, output_height, output_width, 1]))
    # loss and optimizer
    with tf.name_scope('loss'):
      # analyse individual loss
      coordinate_loss = lambda_coord * tf.reduce_sum(target_class *
        tf.square(coordinate - target_coordinate))
      dimension_loss = lambda_coord * tf.reduce_sum(target_class *
        tf.square(dimension - target_dimension))
      class_loss = tf.reduce_sum(target_class *
        tf.square(predict_class - target_class))
      noobj_loss = lambda_no_obj * tf.reduce_sum((1.0 - target_class) *
        tf.square(predict_class - target_class))
      loss = coordinate_loss + dimension_loss + class_loss + noobj_loss
      tf.summary.scalar(name='coordinate_loss', tensor=coordinate_loss)
      tf.summary.scalar(name='dimension_loss', tensor=dimension_loss)
      tf.summary.scalar(name='class_loss', tensor=class_loss)
      tf.summary.scalar(name='noobj_loss', tensor=noobj_loss)
      tf.summary.scalar(name='loss', tensor=loss)
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      train_op = optimizer.minimize(loss)
    return inputs, target_coordinate, target_dimension, target_class, \
      keep_prob, coordinate, dimension, predict_class, loss, train_op

  def load(self, sess, checkpoint_path):
    if os.path.isfile(checkpoint_path + '.meta') and \
      os.path.isfile(checkpoint_path + '.index'):
      self.logger.info('loading last %s checkpoint...' % (checkpoint_path))
      self.saver.restore(sess, checkpoint_path)
      self.start_epoch = int(checkpoint_path.split('-')[-1].strip())
    else:
      self.logger.warning('%s does not exists' % (checkpoint_path))

  def train(self, sess, images, coordinates, dimensions, target_class,
      batch_size=64, output_period=1000, max_epoch=10000,
      decay_epoch=1000, keep_prob=0.8):
    # initialize
    if self.start_epoch == 0:
      self.logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())
    # training
    self.logger.info('start training...')
    last = time.time()
    for epoch in range(self.start_epoch, self.start_epoch + max_epoch + 1):
      offset = (epoch * batch_size) % (images.shape[0] - batch_size + 1)
      batch_data = images[offset:offset+batch_size, :]
      batch_coordinate = coordinates[offset:offset+batch_size, :]
      batch_dimension = dimensions[offset:offset+batch_size, :]
      batch_class = target_class[offset:offset+batch_size, :]
      _, loss = sess.run([self.train_op, self.loss], feed_dict={
        self.inputs: batch_data,
        self.target_coordinate: batch_coordinate,
        self.target_dimension: batch_dimension,
        self.target_class: batch_class,
        self.keep_prob: keep_prob
      })
      if epoch % output_period == 0:
        ms, loss = sess.run([self.merged_summary, self.loss], feed_dict={
          self.inputs: batch_data,
          self.target_coordinate: batch_coordinate,
          self.target_dimension: batch_dimension,
          self.target_class: batch_class,
          self.keep_prob: 1.0
        })
        self.logger.info('%d. loss: %f | time used: %f | saving...' %
            (epoch, loss, (time.time() - last) / output_period))
        self.summary_writer.add_summary(ms, global_step=epoch)
        # save checkpoint
        self.saver.save(sess, self.checkpoint_path, global_step=epoch)
        last = time.time()
      if epoch % decay_epoch == 0 and epoch > 0:
        self.logger.info('decay learning rate...')
        sess.run(self.decay_lr)

  def predict(self, sess, images):
    feed_dict={
      self.inputs: images,
      self.keep_prob: 1.0
    }
    return sess.run([self.coordinate, self.dimension, self.predict_class],
      feed_dict)


def test():
  batch_size = 10
  output_size = 10
  input_width = 160
  input_height = 120
  test_data = np.random.randn(batch_size, input_height, input_width, 3) * \
    127 + 128
  test_coordinate = np.random.randn(batch_size, output_size, output_size, 2)
  test_dimension = np.random.randn(batch_size, output_size, output_size, 2)
  test_dimension[test_dimension <= 0] = 0.0
  test_class = np.random.randn(batch_size, output_size, output_size, 1)
  test_class[test_class > 0] = 1.0
  test_class[test_class <= 0] = 0.0

  yolo = YOLOModel(input_width, input_height, 3, 10, 10, 1)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    yolo.train(sess, test_data, test_coordinate, test_dimension, test_class,
        batch_size=10, output_period=3, max_epoch=30)


if __name__ == '__main__':
  test()

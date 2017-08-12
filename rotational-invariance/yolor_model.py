import tensorflow as tf
import numpy as np
import os
import logging
import time


class YOLORotationModel(object):
  def __init__(self, input_width, input_height, input_channel,
        output_width, output_height, bounding_box, num_classes,
        model_name='YOLORotationModel',
        batch_size=256, learning_rate=1e-3, decay=0.9, saving=True):
    # logger
    logging.basicConfig()
    self.logger = logging.getLogger(model_name)
    self.logger.setLevel(logging.INFO)
    self.logger.info('setting up model...')
    # model
    with tf.variable_scope(model_name):
      self.inputs, \
        self.class_labels, \
        self.xy_labels, \
        self.size_labels, \
        self.rotation_labels, \
        self.indicator_labels, \
        self.keep_prob, \
        self.class_outputs, \
        self.class_prob_outputs, \
        self.xy_outputs, \
        self.size_outputs, \
        self.rotation_outputs, \
        self.loss, \
        self.train_op = \
        self._build_model(input_width, input_height, input_channel,
        output_width, output_height, bounding_box, num_classes,
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

  def _build_model(self,
      input_width, input_height, input_channel,
      output_width, output_height, bounding_box, num_classes,
      learning_rate, decay,
      lambda_coord=2.0, lambda_class=1.2, lambda_noobj=0.1,
      lambda_rotation=1.0):
    """
    inputs:
      rgb image

    outputs:
      class
      xy coordinates
      size
      rotation
    """
    # inputs
    with tf.device('/cpu:0'):
      inputs = tf.placeholder(dtype=tf.float32,
        shape=[None, input_height, input_width, input_channel],
        name='input_images')
      indicator_labels = tf.placeholder(dtype=tf.float32,
        shape=[None, output_height, output_width, bounding_box],
        name='indicator_labels')
      xy_labels = tf.placeholder(dtype=tf.float32,
        shape=[None, output_height, output_width, bounding_box * 2],
        name='xy_labels')
      size_labels = tf.placeholder(dtype=tf.float32,
        shape=[None, output_height, output_width, bounding_box * 2],
        name='size_labels')
      rotation_labels = tf.placeholder(dtype=tf.float32,
        shape=[None, output_height, output_width, bounding_box],
        name='rotation_labels')
      class_labels = tf.placeholder(dtype=tf.float32,
        shape=[None, output_height, output_width, num_classes],
        name='class_labels')
      keep_prob = tf.placeholder(dtype=tf.float32, shape=(),
        name='keep_prob')
    tf.summary.image(name='input_images', tensor=inputs)
    tf.summary.image(name='indicator_labels', tensor=indicator_labels)
    tf.summary.histogram(name='xy_labels', values=xy_labels)
    tf.summary.histogram(name='size_labels', values=size_labels)
    tf.summary.histogram(name='rotation_labels', values=rotation_labels)
    # learning rate
    with tf.device('/cpu:0'):
      self.learning_rate = tf.Variable(learning_rate, name='learning_rate',
        trainable=False)
      self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * decay)
    tf.summary.scalar(name='learning_rate', tensor=self.learning_rate)
    # model
    with tf.name_scope('conv1'):
      h1_size = 32
      with tf.device('/cpu:0'):
        w = tf.get_variable(name='conv_w1',
          shape=[7, 7, input_channel, h1_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.05))
        b = tf.get_variable(name='conv_b1', shape=[h1_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      with tf.device('/gpu:0'):
        h = tf.nn.relu(tf.nn.conv2d(inputs / 255.0, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
        h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
          padding='SAME')
        h = tf.nn.dropout(h, keep_prob)
      tf.summary.histogram(name='conv1_output', values=h)
    with tf.name_scope('conv2'):
      h2_size = 32
      with tf.device('/cpu:0'):
        w = tf.get_variable(name='conv_w2',
          shape=[5, 5, h1_size, h2_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.05))
        b = tf.get_variable(name='conv_b2', shape=[h2_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      with tf.device('/gpu:0'):
        h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
        h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
          padding='SAME')
        h = tf.nn.dropout(h, keep_prob)
      tf.summary.histogram(name='conv2_output', values=h)
    with tf.name_scope('conv3'):
      h3_size = 64
      with tf.device('/cpu:0'):
        w = tf.get_variable(name='conv_w3',
          shape=[5, 5, h2_size, h3_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.05))
        b = tf.get_variable(name='conv_b3', shape=[h3_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      with tf.device('/gpu:0'):
        h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
        h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
          padding='SAME')
        h = tf.nn.dropout(h, keep_prob)
      tf.summary.histogram(name='conv3_output', values=h)
    with tf.name_scope('conv4'):
      h4_size = 64
      with tf.device('/cpu:0'):
        w = tf.get_variable(name='conv_w4',
          shape=[3, 3, h3_size, h4_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.06))
        b = tf.get_variable(name='conv_b4', shape=[h4_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      with tf.device('/gpu:0'):
        h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
        h = tf.nn.dropout(h, keep_prob)
      tf.summary.histogram(name='conv4_output', values=h)
    with tf.name_scope('conv5'):
      h5_size = 64
      with tf.device('/cpu:0'):
        w = tf.get_variable(name='conv_w5',
          shape=[3, 3, h4_size, h5_size],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=0.06))
        b = tf.get_variable(name='conv_b5', shape=[h5_size],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
      with tf.device('/gpu:0'):
        h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
          padding='SAME') + b)
        h = tf.nn.dropout(h, keep_prob)
      tf.summary.histogram(name='conv5_output', values=h)
    with tf.name_scope('fc6'):
      h_shape = h.get_shape().as_list()
      connect_size = h_shape[1] * h_shape[2] * h_shape[3]
      h6_size = 4096
      w = tf.get_variable(name='fc_w6',
        shape=[connect_size, h6_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / connect_size)))
      b = tf.get_variable(name='fc_b6', shape=[h6_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.matmul(tf.reshape(h, [-1, connect_size]), w) + b)
      tf.summary.histogram(name='fc6_output', values=h)
    with tf.name_scope('output'):
      with tf.name_scope('class_probability'):
        with tf.device('/cpu:0'):
          class_prob_output_size = output_width * output_height * bounding_box
          w = tf.get_variable(name='class_prob_ow',
            shape=[h6_size, class_prob_output_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0 / h6_size)))
          b = tf.get_variable(name='class_prob_ob',
            shape=[class_prob_output_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        with tf.device('/gpu:0'):
          class_prob_logits = tf.matmul(h, w) + b
          class_prob_outputs = tf.reshape(class_prob_logits,
            [-1, output_height, output_width, bounding_box],
            name='class_prob_ouputs')
        tf.summary.image(name='class_prob_outputs', tensor=class_prob_outputs)
      with tf.name_scope('xy_output'):
        with tf.device('/cpu:0'):
          xy_output_size = output_width * output_height * bounding_box * 2
          w = tf.get_variable(name='xy_ow',
            shape=[h6_size, xy_output_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0 / h6_size)))
          b = tf.get_variable(name='xy_ob',
            shape=[xy_output_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        with tf.device('/gpu:0'):
          xy_logits = tf.reshape(tf.matmul(h, w) + b,
            [-1, output_height, output_width, bounding_box * 2])
          xy_outputs = tf.multiply(xy_logits,
            tf.constant([input_width, input_height], tf.float32),
            name='xy_outputs')
        tf.summary.histogram(name='xy_outputs', values=xy_outputs)
      with tf.name_scope('size_output'):
        with tf.device('/cpu:0'):
          size_output_size = output_width * output_height * bounding_box * 2
          w = tf.get_variable(name='size_ow',
            shape=[h6_size, size_output_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0 / h6_size)))
          b = tf.get_variable(name='size_ob',
            shape=[size_output_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        with tf.device('/gpu:0'):
          size_logits = tf.reshape(tf.matmul(h, w) + b,
            [-1, output_height, output_width, bounding_box * 2])
          size_outputs = tf.multiply(size_logits,
            tf.constant([input_width, input_height], tf.float32),
            name='size_outputs')
        tf.summary.histogram(name='size_outputs', values=size_outputs)
      with tf.name_scope('rotation_output'):
        with tf.device('/cpu:0'):
          rotation_output_size = output_width * output_height * bounding_box
          w = tf.get_variable(name='rotation_ow',
            shape=[h6_size, rotation_output_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0 / h6_size)))
          b = tf.get_variable(name='rotation_ob',
            shape=[rotation_output_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        with tf.device('/gpu:0'):
          rotation_logits = tf.reshape(tf.matmul(h, w) + b,
            [-1, output_height, output_width, bounding_box],
            name='rotation_output')
          rotation_outputs = rotation_logits
        tf.summary.histogram(name='rotation_outputs', values=rotation_outputs)
      with tf.name_scope('classes'):
        with tf.device('/cpu:0'):
          class_output_size = output_width * output_height * num_classes
          w = tf.get_variable(name='class_ow',
            shape=[h6_size, class_output_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0 / h6_size)))
          b = tf.get_variable(name='class_ob',
            shape=[class_output_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
        with tf.device('/gpu:0'):
          class_logits = tf.matmul(h, w) + b
          class_outputs = tf.reshape(class_logits,
            [-1, output_height, output_width, num_classes],
            name='class_ouputs')
        tf.summary.histogram(name='class_outputs', values=class_outputs)
    # loss and optimizer
    with tf.name_scope('loss'):
      with tf.device('/gpu:0'):
        batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
        # leaky
      with tf.name_scope('class_loss'):
        with tf.device('/gpu:0'):
          class_loss = lambda_class * tf.reduce_sum(indicator_labels *
            tf.square(class_outputs - class_labels)) / batch_size / num_classes
          class_accuracy = tf.reduce_sum(
            tf.reshape(indicator_labels, [-1, output_height, output_width]) *
            tf.cast(tf.equal(tf.argmax(class_outputs, axis=3),
              tf.argmax(class_labels, axis=3)), tf.float32)) * \
                100.0 / tf.reduce_sum(indicator_labels)

          class_prob_loss = lambda_class * tf.reduce_sum(
            indicator_labels * tf.square(class_prob_outputs -
            indicator_labels)) / batch_size
          noobject_loss = lambda_noobj * tf.reduce_sum(
            (1.0 - indicator_labels) * tf.square(class_prob_outputs -
            indicator_labels)) / batch_size
          class_prob_error = tf.reduce_sum(
            tf.abs(class_prob_outputs - indicator_labels)) / batch_size
        tf.summary.scalar(name='class_loss', tensor=class_loss)
        tf.summary.scalar(name='noobject_loss', tensor=noobject_loss)
        tf.summary.scalar(name='class_prob_loss', tensor=class_prob_loss)
        tf.summary.scalar(name='class_accuracy', tensor=class_accuracy)
        tf.summary.scalar(name='class_prob_error', tensor=class_prob_error)
      with tf.name_scope('xy_loss'):
        with tf.device('/gpu:0'):
          xy_loss = lambda_coord * tf.reduce_sum(
            indicator_labels * tf.square(xy_logits - xy_labels)) / batch_size
          xy_error = tf.reduce_sum(
            indicator_labels * tf.abs(xy_logits - xy_labels)) / batch_size
        tf.summary.scalar(name='xy_loss', tensor=xy_loss)
        tf.summary.scalar(name='xy_error', tensor=xy_error)
      with tf.name_scope('size_loss'):
        with tf.device('/gpu:0'):
          size_loss = lambda_coord * tf.reduce_sum(indicator_labels *
            tf.square(size_logits - size_labels)) / batch_size
          size_error = tf.reduce_sum(indicator_labels *
            tf.abs(size_logits - size_labels)) / batch_size
        tf.summary.scalar(name='size_loss', tensor=size_loss)
        tf.summary.scalar(name='size_error', tensor=size_error)
      with tf.name_scope('rotation_loss'):
        with tf.device('/gpu:0'):
          rotation_loss = lambda_rotation * tf.reduce_sum(indicator_labels *
            tf.square(rotation_logits - rotation_labels)) / \
            360.0 / batch_size
          rotation_error = tf.reduce_sum(indicator_labels *
            tf.abs(rotation_logits - rotation_labels)) / batch_size
        tf.summary.scalar(name='rotation_loss', tensor=rotation_loss)
        tf.summary.scalar(name='rotation_error', tensor=rotation_error)
      with tf.device('/gpu:0'):
        loss = class_loss + class_prob_loss + noobject_loss + \
          xy_loss + size_loss + rotation_loss
      tf.summary.scalar(name='total_loss', tensor=loss)
    with tf.name_scope('optimizer'):
      with tf.device('/gpu:0'):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)
    return inputs, \
        class_labels, \
        xy_labels, \
        size_labels, \
        rotation_labels, \
        indicator_labels, \
        keep_prob, \
        class_outputs, \
        class_prob_outputs, \
        xy_outputs, \
        size_outputs, \
        rotation_outputs, \
        loss, train_op

  def load(self, sess, checkpoint_path):
    if os.path.isfile(checkpoint_path + '.meta') and \
        os.path.isfile(checkpoint_path + '.index'):
      self.logger.info('loading last %s checkpoint...' % (checkpoint_path))
      self.saver.restore(sess, checkpoint_path)
      self.start_epoch = int(checkpoint_path.split('-')[-1].strip())
    else:
      self.logger.warning('%s does not exists' % (checkpoint_path))

  def train_batch(self,
      sess,
      batch_images,
      batch_indicator_labels,
      batch_xy_labels,
      batch_size_labels,
      batch_rotation_labels,
      batch_class_labels,
      keep_prob):
    _, loss = sess.run([self.train_op, self.loss], feed_dict={
      self.inputs: batch_images,
      self.indicator_labels: batch_indicator_labels,
      self.xy_labels: batch_xy_labels,
      self.size_labels: batch_size_labels,
      self.rotation_labels: batch_rotation_labels,
      self.class_labels: batch_class_labels,
      self.keep_prob: keep_prob
    })
    return loss

  def train(self,
      sess,
      images,
      indicator_labels,
      xy_labels,
      size_labels,
      rotation_labels,
      class_labels,
      batch_size=256,
      output_period=1000,
      keep_prob=0.8,
      max_epoch=10000,
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
      indexes = np.random.permutation(np.arange(len(images)))[:batch_size]
      batch_images = images[indexes, :]
      batch_indicator_labels = indicator_labels[indexes, :]
      batch_xy_labels = xy_labels[indexes, :]
      batch_size_labels = size_labels[indexes, :]
      batch_rotation_labels = rotation_labels[indexes, :]
      batch_class_labels = class_labels[indexes, :]
      loss = self.train_batch(sess,
        batch_images, batch_indicator_labels, batch_xy_labels,
        batch_size_labels, batch_rotation_labels, batch_class_labels, keep_prob)
      # output
      if epoch % output_period == 0:
        feed_dict = {
          self.inputs: batch_images,
          self.indicator_labels: batch_indicator_labels,
          self.xy_labels: batch_xy_labels,
          self.size_labels: batch_size_labels,
          self.rotation_labels: batch_rotation_labels,
          self.class_labels: batch_class_labels,
          self.keep_prob: 1.0
        }
        ms, loss = sess.run(
          [self.merged_summary, self.loss], feed_dict)
        self.logger.info('%d. loss: %f | time used: %f' %
          (epoch, loss, (time.time() - last)))
        last = time.time()
        # save
        if self.saving:
          self.saver.save(sess, self.checkpoint_path, global_step=epoch)
          self.summary_writer.add_summary(ms, global_step=epoch)
      # learning rate decay
      if epoch % decay_epoch == 0 and epoch != 0:
        self.logger.info('decay learning rate...')
        sess.run(self.decay_lr)

  def predict(self, sess, data):
    return sess.run(self.outputs, feed_dict={
      self.inputs: data,
      self.keep_prob: 1.0
    })


def test():
  input_width, input_height, input_channel = 160, 120, 3
  output_width, output_height, bounding_box, num_classes = 16, 12, 1, 10

  # prepare fake data
  test_batch_size = 1024
  data = np.random.randn(test_batch_size,
    input_height, input_width, input_channel)
  label = np.zeros(shape=[test_batch_size,
    output_height, output_width, bounding_box * 6 + num_classes])
  label[:, :, :, 0] = 1

  model = YOLORotationModel(input_width, input_height, input_channel,
    output_width, output_height, bounding_box, num_classes,
    model_name='test', saving=False)

  with tf.Session() as sess:
    model.train(sess,
      data,
      label[:, :, :, 0:1],
      label[:, :, :, 1:3],
      label[:, :, :, 3:5],
      label[:, :, :, 5:6],
      label[:, :, :, 6:],
      batch_size=32,
      output_period=10,
      keep_prob=0.8,
      max_epoch=100)


if __name__ == '__main__':
  test()

import tensorflow as tf
import numpy as np
import logging
import os
import time
from data_util.tiny_yolo_util import NUM_OUTPUT_CLASSES, NUM_BOUNDING_BOX
from data_util.tiny_yolo_util import OUTPUT_WIDTH, OUTPUT_HEIGHT
from data_util.tiny_yolo_util import TinyYOLODataBatch


class TinyYOLOV2(object):
  def __init__(self, input_width, input_height, input_channel,
      model_name='TinyYOLOV2',
      learning_rate=1e-3, decay=0.9, saving=True,
      lambda_coord=6.0, lambda_noobj=0.6, lambda_class=1.2):
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
          learning_rate, decay, lambda_coord, lambda_noobj, lambda_class)
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
      learning_rate, decay, lambda_coord, lambda_noobj, lambda_class):
    # inputs
    with tf.device('/cpu:0'):
      inputs = tf.placeholder(dtype=tf.float32,
        shape=[None, input_height, input_width, input_channel],
        name='input_images')
      labels = tf.placeholder(dtype=tf.float32, shape=[None,
        OUTPUT_HEIGHT, OUTPUT_WIDTH,
        NUM_OUTPUT_CLASSES + NUM_BOUNDING_BOX * 5], name='labels')
      keep_prob = tf.placeholder(dtype=tf.float32, shape=(),
        name='keep_prob')
      self.global_step = tf.contrib.framework.get_or_create_global_step()
    tf.summary.image(name='input_images', tensor=inputs)
    class_label, boxes = tf.split(labels, [NUM_OUTPUT_CLASSES,
      5 * NUM_BOUNDING_BOX], axis=3)
    self.logger.info('class label shape: %s' % (str(class_label)))
    self.logger.info('box shape: %s' % (str(boxes)))
    box_split = []
    for _ in range(NUM_BOUNDING_BOX):
      box_split.append(1)
      box_split.append(2)
      box_split.append(2)
    roi = tf.split(boxes, box_split, axis=3)
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
        initializer=tf.random_normal_initializer(stddev=0.006))
      b = tf.get_variable(name='conv_b1', shape=[h1_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.conv2d(inputs / 128.0, w, strides=[1, 1, 1, 1],
        padding='SAME') + b
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.relu(h)
      tf.summary.histogram('conv1', values=h)
      #  h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv2'):
      h2_size = 64
      w = tf.get_variable(name='conv_w2',
        shape=[5, 5, h1_size, h2_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.025))
      b = tf.get_variable(name='conv_b2', shape=[h2_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.relu(h)
      tf.summary.histogram('conv2', values=h)
      #  h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('conv3'):
      h3_size = 256
      w = tf.get_variable(name='conv_w3',
        shape=[5, 5, h2_size, h3_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.017))
      b = tf.get_variable(name='conv_b3', shape=[h3_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b
      h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
        padding='SAME')
      h = tf.nn.relu(h)
      tf.summary.histogram('conv3', values=h)
      h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('fc4'):
      h_shape = h.get_shape().as_list()
      connect_size = h_shape[1] * h_shape[2] * h_shape[3]
      h4_size = 1024
      w = tf.get_variable(name='w4',
        shape=[connect_size, h4_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / connect_size)))
      b = tf.get_variable(name='b4', shape=[h4_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-3))
      h = tf.nn.relu(tf.matmul(tf.reshape(h, [-1, connect_size]), w) + b)
      tf.summary.histogram('fc4', values=h)
    output_connect = h4_size
    with tf.name_scope('output'):
      with tf.name_scope('class'):
        class_output = NUM_OUTPUT_CLASSES * OUTPUT_HEIGHT * OUTPUT_WIDTH
        w = tf.get_variable(name='class_ow',
          shape=[output_connect, class_output],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / output_connect)))
        b = tf.get_variable(name='class_ob', shape=[class_output],
          dtype=tf.float32,
          initializer=tf.constant_initializer(value=1e-3))
        class_logits = tf.matmul(h, w) + b
        class_outputs = tf.reshape(class_logits,
          [-1, OUTPUT_HEIGHT, OUTPUT_WIDTH, NUM_OUTPUT_CLASSES])
        tf.summary.histogram(name='class_outputs', values=class_outputs)
      with tf.name_scope('indicator'):
        indicators = []
        indicator_size = OUTPUT_HEIGHT * OUTPUT_WIDTH
        for i in range(NUM_BOUNDING_BOX):
          w = tf.get_variable(name='indicator%d_ow' % (i),
            shape=[output_connect, indicator_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0 / output_connect)))
          b = tf.get_variable(name='ndicator%d_ob' % (i),
            shape=[indicator_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
          ind = tf.reshape(tf.matmul(h, w) + b,
            [-1, OUTPUT_HEIGHT, OUTPUT_WIDTH, 1])
          self.logger.info('Indicator %d. shape: %s' % (i, str(ind)))
          indicators.append(ind)
          ind_image = tf.reshape(ind, [-1, OUTPUT_HEIGHT, OUTPUT_WIDTH, 1])
          tf.summary.image(name='indicator%d_outputs' % (i), tensor=ind_image)
      with tf.name_scope('xy'):
        xys = []
        xy_size = OUTPUT_HEIGHT * OUTPUT_WIDTH * 2
        for i in range(NUM_BOUNDING_BOX):
          w = tf.get_variable(name='xy%d_ow' % (i),
            shape=[output_connect, xy_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0 / output_connect)))
          b = tf.get_variable(name='xy%d_ob' % (i),
            shape=[xy_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
          xy = tf.reshape(tf.matmul(h, w) + b,
            [-1, OUTPUT_HEIGHT, OUTPUT_WIDTH, 2])
          self.logger.info('xy %d. xy shape: %s' % (i, str(xy)))
          xys.append(xy)
          tf.summary.histogram(name='xy%d_outputs' % (i), values=xy)
      with tf.name_scope('size'):
        sizes = []
        for i in range(NUM_BOUNDING_BOX):
          w = tf.get_variable(name='size%d_ow' % (i),
            shape=[output_connect, xy_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0 / output_connect)))
          b = tf.get_variable(name='size%d_ob' % (i),
            shape=[xy_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=1e-3))
          size = tf.reshape(tf.matmul(h, w) + b,
            [-1, OUTPUT_HEIGHT, OUTPUT_WIDTH, 2])
          self.logger.info('size %d. size shape: %s' % (i, str(size)))
          sizes.append(size)
          tf.summary.histogram(name='size%d_outputs' % (i), values=size)
      output = [class_outputs]
      for i in range(NUM_BOUNDING_BOX):
        output.append(indicators[i])
        output.append(xys[i])
        output.append(sizes[i])
      outputs = tf.concat(output, axis=3)
      self.logger.info('output shape: %s' % (str(outputs)))
    with tf.name_scope('loss'):
      batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
      valid = tf.cast(tf.equal(roi[0], 1.0), tf.float32)
      self.logger.info('valid grid cell: %s' % (str(valid)))
      with tf.name_scope('class'):
        class_diff = class_outputs - class_label
        class_loss = tf.reduce_sum(tf.square(valid * class_diff)) / batch_size
        class_accuracy = tf.reduce_sum(valid *
          tf.cast(tf.expand_dims(tf.equal(tf.argmax(class_outputs, axis=3),
            tf.argmax(class_label, axis=3)), axis=-1), tf.float32)) * 100.0 / \
              tf.reduce_sum(valid)
        tf.summary.scalar('class_loss', tensor=class_loss)
        tf.summary.scalar('class_accuracy', tensor=class_accuracy)
      with tf.name_scope('indicator'):
        indicator_loss = 0
        no_obj_loss = 0
        for i in range(NUM_BOUNDING_BOX):
          self.logger.info('indicator: %s | target: %s' %
            (str(indicators[i * 5]), str(roi[i * 3])))
          indicator_loss += tf.reduce_sum(valid *
            tf.square(indicators[i * 5] - roi[i * 3])) / batch_size
          no_obj_loss += tf.reduce_sum((1.0 - valid) *
            tf.square(indicators[i * 5] - roi[i * 3])) / batch_size
          tf.summary.image(name='indicator%d_target' % (i), tensor=roi[i * 3])
        tf.summary.scalar('indicator_loss', tensor=indicator_loss)
        tf.summary.scalar('no_obj_loss', tensor=no_obj_loss)
      with tf.name_scope('xy'):
        xy_loss = 0
        for i in range(NUM_BOUNDING_BOX):
          self.logger.info('xy: %s | target: %s' %
            (str(xys[i * 5]), str(roi[i * 3 + 1])))
          xy_loss += tf.reduce_sum(valid *
            tf.square(xys[i * 5] - roi[i * 3 + 1])) / batch_size
        tf.summary.scalar('xy_loss', tensor=xy_loss)
      with tf.name_scope('size'):
        size_loss = 0
        for i in range(NUM_BOUNDING_BOX):
          self.logger.info('sizes: %s | target: %s' %
            (str(sizes[i * 5]), str(roi[i * 3 + 2])))
          size_loss += tf.reduce_sum(valid *
            tf.square(sizes[i * 5] - roi[i * 3 + 2])) / batch_size
        tf.summary.scalar('size_loss', tensor=size_loss)
      total_loss = lambda_class * class_loss + \
        lambda_class * indicator_loss + \
        lambda_noobj * no_obj_loss + \
        lambda_coord * xy_loss + lambda_coord * size_loss
      tf.summary.scalar('total_loss', tensor=total_loss)
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      train_op = optimizer.minimize(total_loss)
    return inputs, labels, keep_prob, outputs, total_loss, class_accuracy, train_op

  def load(self, sess, checkpoint_path):
    if os.path.isfile(checkpoint_path + '.meta') and \
        os.path.isfile(checkpoint_path + '.index'):
      self.logger.info('loading last %s checkpoint...' % (checkpoint_path))
      self.saver.restore(sess, checkpoint_path)
      self.start_epoch = int(checkpoint_path.split('-')[-1].strip())
    else:
      self.logger.warning('%s does not exists' % (checkpoint_path))

  def train_batch(self, sess, batch_data, batch_label, keep_prob):
    _, loss = sess.run([self.train_op, self.loss], feed_dict={
      self.inputs: batch_data,
      self.labels: batch_label,
      self.keep_prob: keep_prob
    })
    return loss

  def train(self, sess, batcher,
      batch_size=256, output_period=100,
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
      batch_data, batch_label = batcher.batch()
      loss = self.train_batch(sess, batch_data, batch_label, keep_prob)
      # output
      if epoch % output_period == 0:
        feed_dict = {
          self.inputs: batch_data,
          self.labels: batch_label,
          self.keep_prob: 1.0
        }
        ms, loss, accuracy = sess.run(
          [self.merged_summary, self.loss, self.accuracy], feed_dict)
        self.logger.info('%d. loss: %f | accuracy: %f | time used: %f' %
          (epoch, loss, accuracy, (time.time() - last)))
        last = time.time()
        # save
        if self.saving:
          self.saver.save(sess, self.checkpoint_path,
            global_step=self.global_step)
          self.summary_writer.add_summary(ms,
            global_step=self.global_step)
      # learning rate decay
      if epoch % decay_epoch == 0 and epoch != 0:
        sess.run(self.decay_lr)

  def predict(self, sess, image):
    return sess.run(self.outputs, feed_dict={
      self.inputs: image,
      self.keep_prob: 1.0
    })


def test():
  input_width, input_height, input_channel = 80, 60, 1

  model = TinyYOLOV2(input_width, input_height, input_channel,
    model_name='test', saving=False)
  batcher = TinyYOLODataBatch('data_util/tiny_yolo_v2.sqlite3', 'blocks',
    input_width, input_height)
  with tf.Session() as sess:
    model.train(sess, batcher,
      batch_size=256,
      output_period=10,
      keep_prob=0.8,
      max_epoch=100)


if __name__ == '__main__':
  test()

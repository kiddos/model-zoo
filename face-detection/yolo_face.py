from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging
import sys
import os
import time
from PIL import Image, ImageDraw
from argparse import ArgumentParser

from wider_loader import WIDERLoader

logging.basicConfig()
logger = logging.getLogger('yolo')
logger.setLevel(logging.INFO)

class YOLOFace(object):
  def __init__(self, input_size, output_size, inference,
      lambda_coord=1.0, lambda_size=10.0, lambda_no_obj=0.86, learning_rate=1e-4):
    self.input_size = input_size
    self.output_size = output_size
    self.output_channel = 1 * 5
    self.lambda_coord = lambda_coord
    self.lambda_size = lambda_size
    self.lambda_no_obj = lambda_no_obj

    if hasattr(self, inference):
      self.inference_func = getattr(self, inference)
      self.inference = inference
    else:
      logger.error('no %s inference function is found.')
      sys.exit(1)

    #  with tf.device('/:cpu0'):
    self._setup_inputs()

    with tf.variable_scope('yolo'):
      logits = self.inference_func(self.input_images)
      ind, coord, s = tf.split(logits, [1, 2, 2], axis=3)

      ind_output = tf.nn.sigmoid(ind)
      input_size = tf.constant(float(self.input_size))
      coord_output = coord * input_size
      size_output = s * input_size
      self.output = tf.concat([ind_output, coord_output, size_output], axis=3)

    with tf.name_scope('loss'):
      indicator, coordinate, size = tf.split(self.label_grids,
        [1, 2, 2], axis=3)
      tf.summary.image('indicator_prediction', ind)

      num_obj = tf.reduce_sum(indicator)
      with tf.name_scope('indicator'):
        indicator_error = ind - indicator
        no_obj = 1.0 - indicator
        num_empty_grid = tf.reduce_sum(no_obj)
        square_error = tf.square(indicator_error)
        self.ind_loss = tf.reduce_sum(indicator * square_error) / num_obj
        self.no_obj_loss = tf.reduce_sum(no_obj * square_error) / num_empty_grid
        tf.summary.scalar('indicator_loss', self.ind_loss)
        tf.summary.scalar('no_obj_loss', self.no_obj_loss)

      with tf.name_scope('coordinate'):
        coord_error = coord - coordinate
        self.coord_loss = tf.reduce_sum(
          indicator * tf.square(coord_error)) / num_obj

        tf.summary.scalar('coord_loss', self.coord_loss)
        tf.summary.scalar('coord_error',
          tf.reduce_sum(indicator * tf.abs(coord_error)) / num_obj)

      with tf.name_scope('size'):
        size_error = s - size
        self.size_loss = tf.reduce_sum(
          indicator * tf.square(size_error)) / num_obj
        tf.summary.scalar('size_loss', self.size_loss)
        tf.summary.scalar('size_error',
          tf.reduce_sum(indicator * tf.abs(size_error)) / num_obj)

      self.loss = self.ind_loss + \
          self.lambda_no_obj * self.no_obj_loss + \
          self.lambda_coord * self.coord_loss + \
          self.lambda_size * self.size_loss
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimization'):
      self.learning_rate = tf.Variable(learning_rate,
        trainable=False, name='learning_rate')
      self.decay_learning_rate = tf.assign(self.learning_rate,
        self.learning_rate * 0.9)
      tf.summary.scalar('learning_rate', self.learning_rate)

      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

    with tf.name_scope('evaluation'):
      batch_size = tf.gather(tf.shape(indicator), 0)
      num_grids = tf.cast(batch_size * (self.output_size ** 2), tf.float32)
      certain = tf.cast(tf.greater_equal(ind_output, 0.5), tf.float32)
      self.indicator_accuracy = tf.reduce_sum(
        indicator * tf.cast(tf.equal(certain, indicator), tf.float32)) / num_obj
      tf.summary.scalar('indicator_accuracy', self.indicator_accuracy)

      left_bottom = tf.maximum(coord - s / 2.0, coordinate - size / 2.0)
      right_top = tf.minimum(coord + s / 2.0, coordinate + size / 2.0)
      left, bottom = tf.split(left_bottom, [1, 1], axis=3)
      right, top = tf.split(left_bottom, [1, 1], axis=3)
      valid_intersection = tf.reduce_all(
        tf.greater(right_top, left_bottom), axis=3)
      area = tf.reduce_prod(right_top - left_bottom, axis=3)
      true_area = tf.maximum(tf.reduce_prod(size, axis=3), 1e-5)
      self.intersect_area = tf.reduce_sum(
        tf.cast(valid_intersection, tf.float32) * area / true_area) / num_grids
      tf.summary.scalar('intersect_area_ratio', self.intersect_area)

  def _setup_inputs(self):
    self.input_images = tf.placeholder(dtype=tf.float32,
      shape=[None, self.input_size, self.input_size, 3], name='input_images')
    self.label_grids = tf.placeholder(dtype=tf.float32, name='labels',
      shape=[None, self.output_size, self.output_size, self.output_channel])
    tf.summary.image('input_images', self.input_images)

    # validation
    self.valid_images = tf.placeholder(dtype=tf.float32,
      shape=[None, self.input_size, self.input_size, 3], name='valid_images')
    self.valid_labels = tf.placeholder(dtype=tf.float32, name='valid_labels',
      shape=[None, self.output_size, self.output_size, self.output_channel])
    tf.summary.image('validation images', self.input_images)

    self.keep_prob = tf.placeholder(dtype=tf.float32,
      shape=[], name='keep_prob')

  def inference_v0(self, inputs):
    ksize = 3
    stddev = 0.016
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 64, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 128, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 256, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(pool, 512, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv5'):
      conv = tf.contrib.layers.conv2d(pool, 1024, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('drop5'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      stddev = np.sqrt(2.0 / connect_size)
      fc_size = 1024
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), fc_size,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('output'):
      connect_size = self.output_size * self.output_size * self.output_channel
      stddev = np.sqrt(2.0 / fc_size)
      output = tf.contrib.layers.fully_connected(fc, connect_size,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

      logits = tf.reshape(output,
        [-1, self.output_size, self.output_size, self.output_channel])
    return logits

  def inference_v1(self, inputs):
    ksize = 3
    stddev = 0.016
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 64, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 256, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(pool, 512, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv5'):
      conv = tf.contrib.layers.conv2d(pool, 1024, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('drop5'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('output'):
      logits = tf.contrib.layers.conv2d(drop, 5, stride=1, kernel_size=ksize,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
    return logits

  def inference_v2(self, inputs):
    ksize = 3
    stddev = 0.016
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 16, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=0.0006))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 32, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 64, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(pool, 128, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('pool4'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv5'):
      conv = tf.contrib.layers.conv2d(pool, 256, stride=1, kernel_size=ksize,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))

    with tf.name_scope('drop5'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('output'):
      logits = tf.contrib.layers.conv2d(drop, 5, stride=1, kernel_size=ksize,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
    return logits

  def predict_batch(self, sess, input_images):
    return sess.run(self.output, feed_dict={
      self.input_images: input_images,
      self.keep_prob: 1.0
    })

  def predict(self, sess, input_image):
    return sess.run(self.output, feed_dict={
      self.input_images: np.expand_dims(input_image, axis=0),
      self.keep_prob: 1.0
    })[0]

  def prepare_folder(self):
    index = 0
    folder = 'yolo_face-%s_%d' % (self.inference, index)
    while os.path.isdir(folder):
      index += 1
      folder = 'yolo_face-%s_%d' % (self.inference, index)
    os.mkdir(folder)
    return folder


def train(args):
  loader = WIDERLoader(args.dbname)
  loader.load_data()

  yolo = YOLOFace(loader.get_input_size(), loader.get_output_size(),
    args.inference,
    lambda_coord=args.lambda_coord,
    lambda_size=args.lambda_size,
    lambda_no_obj=args.lambda_no_obj,
    learning_rate=args.learning_rate)

  if args.saving == 'True':
    folder = yolo.prepare_folder()
    checkpoint = os.path.join(folder, 'yolo')
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
      tf.get_default_graph())

  training_data, training_label = loader.get_training_data()
  training_data_size = len(training_data)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_time = 0
    for epoch in range(args.max_epoches + 1):
      offset = epoch % (training_data_size - args.batch_size)
      to = offset + args.batch_size

      training_data_batch = training_data[offset:to, :]
      training_label_batch = training_label[offset:to, :]

      if epoch % args.display_epoches == 0:
        loss, ind_loss, no_obj_loss, coord_loss, size_loss, \
          indicator_accuarcy, intersect_ratio = sess.run([yolo.loss,
            yolo.ind_loss, yolo.no_obj_loss, yolo.coord_loss,
            yolo.size_loss, yolo.indicator_accuracy,
            yolo.intersect_area], feed_dict={
              yolo.input_images: training_data_batch,
              yolo.label_grids: training_label_batch,
              yolo.keep_prob: 1.0,
            })
        logger.info('%d. loss: %f, ind: %f, no-obj: %f, coord: %f, size: %f',
          epoch, loss, ind_loss, no_obj_loss, coord_loss, size_loss)
        logger.info('indicator accuarcy: %f, intersect ratio: %f',
          indicator_accuarcy, intersect_ratio)

        ave = total_time / (epoch + 1)
        time_remaining = (args.max_epoches - epoch) * ave
        days = int(time_remaining / 86400)
        time_remaining %= 86400
        hours = int(time_remaining / 3600)
        time_remaining %= 3600
        minutes = int(time_remaining / 60)
        time_remaining %= 60
        seconds = int(time_remaining)
        logger.info('time remaining: %d days %d hours %d minutes %d seconds' %
          (days, hours, minutes, seconds))

      if epoch % args.save_epoches == 0 and epoch != 0 and args.saving == 'True':
        saver.save(sess, checkpoint, global_step=epoch)

      if epoch % args.summary_epoches == 0 and epoch != 0 and \
          args.saving == 'True':
        summary = sess.run(yolo.summary, feed_dict={
          yolo.input_images: training_data_batch,
          yolo.label_grids: training_label_batch,
          yolo.keep_prob: 1.0,
        })
        summary_writer.add_summary(summary, global_step=epoch)

      start = time.time()
      sess.run(yolo.train_ops, feed_dict={
        yolo.input_images: training_data_batch,
        yolo.label_grids: training_label_batch,
        yolo.keep_prob: args.keep_prob,
      })
      passed = time.time() - start
      total_time += passed

      if epoch % args.decay_epoches == 0 and epoch != 0:
        logger.info('decay learning rate...')
        sess.run(yolo.decay_learning_rate)


def fit(original, size):
  w, h = original.size
  output_image = Image.new('RGB', (size, size))
  if w >= h:
    scale = float(size) / w
    oh = int(h * scale)
    output_image.paste(original.resize((size, oh)),
      (0, (size - oh) / 2))
  else:
    scale = float(size) / h
    ow = int(w * scale)
    output_image.paste(original.resize((ow, size)),
      ((size - ow) / 2, 0))
  return np.array(output_image, np.uint8)

def inference(args):
  loader = WIDERLoader(args.dbname)

  with tf.device('/:gpu0'):
    yolo = YOLOFace(loader.get_input_size(), loader.get_output_size(),
      args.inference)

  #  test_image = np.array(Image.open('test_image.jpg'), dtype=np.float32)

  config = tf.ConfigProto(log_device_placement=True)
  #  config.gpu_options.per_process_gpu_memory_fraction=0.3
  #  config.operation_timeout_in_ms=50000
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    try:
      import cv2
      camera = cv2.VideoCapture(0)

      count = 0
      total_time = 0
      while True:
        _, img = camera.read(0)
        img = fit(Image.fromarray(img), yolo.input_size)

        start = time.time()
        yolo.predict(sess, img)
        passed = time.time() - start
        total_time += passed
        count += 1

        cv2.imshow('Image', img)
        print('\raverage: %f' % (total_time / count), end='')

        key = cv2.waitKey(10)
        if key == ord('q'):
          break

    except Exception as e:
      logger.error(e)
    #  count = 1000
    #  total = 0
    #  for i in range(count):
    #    start = time.time()
    #    result = yolo.predict(sess, test_image)
    #    passed = time.time() - start
    #    total += passed
    #  print(result)
    #  logger.info('time used for 1 frame: %f' % (total / count))


def main():
  parser = ArgumentParser()
  parser.add_argument('--dbname', dest='dbname', type=str,
    default='wider.sqlite3', help='sqlite3 db to load')

  parser.add_argument('--inference', dest='inference', type=str,
    default='inference_v0', help='model version to use')

  parser.add_argument('--mode', dest='mode', type=str,
    default='train', help='mode to run')

  # training parameters
  parser.add_argument('--learning-rate', dest='learning_rate', type=float,
    default=1e-4, help='learning rate for training')
  parser.add_argument('--lambda-coord', dest='lambda_coord', type=float,
    default=1.6, help='coefficient for coordinate loss')
  parser.add_argument('--lambda-size', dest='lambda_size', type=float,
    default=10.0, help='coefficient for size loss')
  parser.add_argument('--lambda-no-obj', dest='lambda_no_obj', type=float,
    default=0.86, help='coefficient for no-obj loss')
  parser.add_argument('--batch-size', dest='batch_size', type=int,
    default=32, help='batch size for training')
  parser.add_argument('--max-epoches', dest='max_epoches', type=int,
    default=100000, help='max epoches to train')
  parser.add_argument('--display-epoches', dest='display_epoches', type=int,
    default=10, help='epoches to display training result')
  parser.add_argument('--save-epoches', dest='save_epoches', type=int,
    default=1000, help='epoches to save training result')
  parser.add_argument('--summary-epoches', dest='summary_epoches', type=int,
    default=10, help='epoches to save training summary')
  parser.add_argument('--decay-epoches', dest='decay_epoches', type=int,
    default=10000, help='epoches to decay learning rate for training')
  parser.add_argument('--keep-prob', dest='keep_prob', type=float,
    default=0.8, help='keep probability for dropout')
  parser.add_argument('--saving', dest='saving', type=str,
    default='False', help='rather to save the training result')

  args = parser.parse_args()

  if args.mode == 'train':
    train(args)
  elif args.mode == 'inference':
    inference(args)


if __name__ == '__main__':
  main()

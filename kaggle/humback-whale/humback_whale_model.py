import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random


class HumbackWhaleModel(object):
  def __init__(self, image_width, image_height, num_classes, learning_rate):
    self.image_width, self.image_height = image_width, image_height
    self.num_classes = num_classes
    self._setup_inputs()

    with tf.variable_scope('inference'):
      logits, self.output = self.inference(self.input_images)

    with tf.name_scope('loss'):
      labels = tf.one_hot(self.label, num_classes, name='label_vec')
      self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,
          labels=labels, name='loss'))
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimizer'):
      self.learning_rate = tf.Variable(learning_rate, name='learning_rate')
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

    with tf.variable_scope('inference', reuse=True):
      _, valid_output = self.inference(self.valid_images)

    with tf.name_scope('evaluation'):
      self.train_accuarcy = self.evaluate(self.output, name='train_accuracy')
      self.valid_accuarcy = self.evaluate(valid_output, name='valid_accuracy')

      tf.summary.scalar('train_accuracy', self.train_accuarcy)
      tf.summary.scalar('valid_accuracy', self.valid_accuarcy)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def _setup_inputs(self):
    self.input_images = tf.placeholder(dtype=tf.float32,
      shape=[None, self.image_height, self.image_width, 1], name='input_images')
    self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
    self.keep_prob = tf.placeholder(dtype=tf.float32,
      shape=(), name='keep_prob')

    self.valid_images = tf.placeholder(dtype=tf.float32,
      shape=[None, self.image_height, self.image_width, 1], name='valid_images')
    self.valid_label = tf.placeholder(dtype=tf.int32,
      shape=[None], name='valid_label')

  def inference(self, inputs):
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, 3, 1,
        weights_initializer=tf.random_normal_initializer(stddev=0.001))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 64, 3, 1,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(pool, 128, 3, 1,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop3'):
      drop = tf.nn.dropout(pool, self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), 256,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('outputs'):
      logits = tf.contrib.layers.fully_connected(fc, self.num_classes,
        activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
      outputs = tf.nn.softmax(logits, name='prediction')

    return logits, outputs

  def evaluate(self, output, name):
    prediction = tf.cast(tf.argmax(output, axis=1), tf.int32)
    equal = tf.cast(tf.equal(prediction, self.label), tf.float32)
    return tf.reduce_mean(equal, name=name)


def main():
  model = HumbackWhaleModel(32, 32, 4000, 1e-3)

  if os.path.isdir('train'):
    images = os.listdir('train')
    image_path = os.path.join('train', random.sample(images, 1)[0])
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize([32, 32], Image.NEAREST)
    img = np.reshape(np.array(img), [1, 32, 32, 1])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      output = sess.run(model.output, feed_dict={
        model.input_images: img,
        model.keep_prob: 1.0
      })
      print('image: %s' % image_path)
      print('output: %s, %d' % (str(output), np.argmax(output)))


if __name__ == '__main__':
  main()

import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random


class HumbackWhaleModel(object):
  def __init__(self, image_width, image_height, num_classes, learning_rate,
      lambda_reg):
    self.image_width, self.image_height = image_width, image_height
    self.num_classes = num_classes
    self._setup_inputs()

    with tf.variable_scope('inference'):
      self.logits, self.output = self.inference(self.input_images)

    var = tf.trainable_variables()
    reg = 0
    for v in var:
      if len(v.get_shape().as_list()) > 1:
        reg += tf.reduce_sum(tf.square(v))

    with tf.name_scope('loss'):
      labels = tf.one_hot(self.label, num_classes, name='label_vec')
      self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          logits=self.logits, labels=labels))
      self.loss = tf.add(self.loss, reg * lambda_reg, name='loss')
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optimizer'):
      self.learning_rate = tf.Variable(learning_rate,
          trainable=False, name='learning_rate')
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

    with tf.variable_scope('inference', reuse=True):
      _, valid_output = self.inference(self.valid_images)

    with tf.name_scope('evaluation'):
      self.train_accuarcy = self.evaluate(self.output, self.label,
        name='train_accuracy')
      self.valid_accuarcy = self.evaluate(valid_output, self.valid_label,
        name='valid_accuracy')

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
      conv = tf.contrib.layers.conv2d(inputs, 64, 3, 1,
        weights_initializer=tf.random_normal_initializer(stddev=0.01))

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = self.stack_conv(pool, 128, 3, 2)

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop2'):
      drop = tf.nn.dropout(pool, self.keep_prob)

    with tf.name_scope('conv3'):
      conv = self.stack_conv(drop, 256, 3, 4)

    with tf.name_scope('pool3'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('drop3'):
      drop = tf.nn.dropout(pool, self.keep_prob)

    with tf.name_scope('conv4'):
      conv = self.stack_conv(drop, 512, 3, 6)
      conv = tf.contrib.layers.conv2d(conv, 64, 3, 1,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('drop4'):
      drop = tf.nn.dropout(pool, self.keep_prob)

    with tf.name_scope('fully_connected'):
      flatten = tf.contrib.layers.flatten(drop)
      fc = tf.contrib.layers.fully_connected(flatten, 512,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('outputs'):
      logits = tf.contrib.layers.fully_connected(fc, self.num_classes,
        activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
      outputs = tf.nn.softmax(logits, name='prediction')

    return logits, outputs

  def stack_conv(self, inputs, size, ksize, stack=1):
    conv = tf.contrib.layers.conv2d(inputs, size, ksize, 1,
      weights_initializer=tf.variance_scaling_initializer())
    for _ in range(stack):
      conv = tf.contrib.layers.conv2d(conv, size / 2, 1, 1,
        weights_initializer=tf.variance_scaling_initializer())
      conv = tf.contrib.layers.conv2d(conv, size, ksize, 1,
        weights_initializer=tf.variance_scaling_initializer())
    return conv

  def evaluate(self, output, label, name):
    prediction = tf.cast(tf.argmax(output, axis=1), tf.int32)
    equal = tf.cast(tf.equal(prediction, label), tf.float32)
    return tf.reduce_mean(equal, name=name)


def main():
  image_size = 64
  model = HumbackWhaleModel(image_size, image_size, 4251, 1e-3, 1e-4)

  if os.path.isdir('train'):
    images = os.listdir('train')
    image_path = os.path.join('train', random.sample(images, 1)[0])
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize([image_size, image_size], Image.NEAREST)
    img = np.reshape(np.array(img), [1, image_size, image_size, 1])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      output, logits = sess.run([model.output, model.logits], feed_dict={
        model.input_images: img,
        model.keep_prob: 1.0
      })
      print('image: %s' % image_path)
      print('output: %s, %d' % (str(output), np.argmax(output)))
      print('logits: %s, %d' % (str(logits), np.argmax(logits)))


if __name__ == '__main__':
  main()

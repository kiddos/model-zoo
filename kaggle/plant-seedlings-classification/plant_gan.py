import tensorflow as tf
import numpy as np
import os
import logging
from argparse import ArgumentParser

from plant_loader import PlantLoader

logging.basicConfig()
logger = logging.getLogger('plant_gan')
logger.setLevel(logging.INFO)


class Generator(object):
  def __init__(self, num_classes, image_size, noise_size=4):
    self.num_classes = num_classes
    self.noise_size = noise_size
    self.image_size = image_size

    self._setup_inputs()

  def _setup_inputs(self):
    self.input_type = tf.placeholder(dtype=tf.float32, name='input_types',
      shape=[None, self.num_classes])
    self.noise = tf.placeholder(dtype=tf.float32, name='noise',
      shape=[None, self.noise_size])

  def inference(self, inputs, noise, trainable=True):
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, self.num_classes])
    noise_reshaped = tf.reshape(noise, [-1, 1, 1, self.noise_size])

    with tf.name_scope('inputs'):
      concat = tf.concat([input_reshaped, noise_reshaped], axis=3)

    with tf.name_scope('conv1'):
      size = self.image_size * self.image_size
      conv = tf.contrib.layers.conv2d(concat, size, stride=1, kernel_size=1,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())
      conv = tf.reshape(conv, [-1, self.image_size, self.image_size, 1])

    with tf.name_scope('norm1'):
      norm = tf.layers.batch_normalization(conv)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(norm, 8, stride=1, kernel_size=3,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv3'):
      conv = tf.contrib.layers.conv2d(conv, 16, stride=1, kernel_size=3,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv4'):
      conv = tf.contrib.layers.conv2d(conv, 32, stride=1, kernel_size=3,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv5'):
      conv = tf.contrib.layers.conv2d(conv, 64, stride=1, kernel_size=3,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('conv6'):
      conv = tf.contrib.layers.conv2d(conv, 128, stride=1, kernel_size=3,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      #  output = tf.contrib.layers.conv2d(conv, 3, stride=1, kernel_size=ksize,
      #    trainable=trainable, activation_fn=None,
      #    weights_initializer=tf.random_normal_initializer(stddev=127.0),
      #    biases_initializer=tf.random_normal_initializer(mean=127.0,
      #      stddev=10.0))
      output = tf.contrib.layers.conv2d(conv, 3, stride=1, kernel_size=3,
        trainable=trainable, activation_fn=None,
        weights_initializer=tf.variance_scaling_initializer())
      #  output = (output * 32 * 127) + 127.0
    return output


class Discriminator(object):
  def __init__(self, num_classes, image_size):
    self.num_classes = num_classes
    self.output_size = num_classes + 1
    self.image_size = image_size

    self._setup_inputs()

  def _setup_inputs(self):
    self.input_images = tf.placeholder(dtype=tf.float32, name='input_images',
      shape=[None, self.image_size, self.image_size, 3])
    self.labels = tf.placeholder(dtype=tf.float32, name='labels',
      shape=[None, self.output_size])
    self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob',
      shape=[])

  def inference(self, inputs, trainable=True):
    ksize = 3
    with tf.name_scope('conv1'):
      conv = tf.contrib.layers.conv2d(inputs, 32, stride=1, kernel_size=ksize,
        trainable=trainable,
        weights_initializer=tf.random_normal_initializer(stddev=1.0))
      #  conv = tf.contrib.layers.conv2d(inputs, 32, stride=1, kernel_size=ksize,
      #    trainable=trainable,
      #    weights_initializer=tf.variance_scaling_initializer())

    #  with tf.name_scope('norm1'):
    #    norm = tf.layers.batch_normalization(conv)

    with tf.name_scope('pool1'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv2'):
      conv = tf.contrib.layers.conv2d(pool, 64, stride=1, kernel_size=ksize,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('pool2'):
      pool = tf.contrib.layers.max_pool2d(conv, 2)

    with tf.name_scope('conv3'):
      #  conv = self.multiple_conv(pool, 64, trainable)
      conv = tf.contrib.layers.conv2d(pool, 128, stride=1, kernel_size=ksize,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('drop3'):
      drop = tf.nn.dropout(conv, keep_prob=self.keep_prob)

    with tf.name_scope('fully_connected'):
      connect_shape = drop.get_shape().as_list()
      connect_size = connect_shape[1] * connect_shape[2] * connect_shape[3]
      fc = tf.contrib.layers.fully_connected(
        tf.reshape(drop, [-1, connect_size]), 1024,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())

    with tf.name_scope('output'):
      logits = tf.contrib.layers.fully_connected(fc, self.output_size,
        activation_fn=None, trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())
      output = tf.nn.softmax(logits)
    return logits, output

  def multiple_conv(self, inputs, size, trainable, multiple=1):
    conv = tf.contrib.layers.conv2d(inputs, size, stride=1, kernel_size=3,
      trainable=trainable,
      weights_initializer=tf.variance_scaling_initializer())
    for i in range(multiple):
      conv = tf.contrib.layers.conv2d(conv, size / 2, stride=1, kernel_size=1,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())
      conv = tf.contrib.layers.conv2d(conv, size, stride=1, kernel_size=3,
        trainable=trainable,
        weights_initializer=tf.variance_scaling_initializer())
    return conv


def norm_images(images):
  normed_images = np.copy(images) / 255.0
  return normed_images


def expand_label(label):
  new_label = np.zeros(shape=[label.shape[0], label.shape[1] + 1],
    dtype=np.float32)
  new_label[:, 1:] = label
  return new_label


class PlantGAN(object):
  def __init__(self, num_classes, input_size,
      glearning_rate, dlearning_rate):
    self.num_classes = num_classes
    self.input_size = input_size

    generator = Generator(self.num_classes, self.input_size)
    with tf.variable_scope('generator_train'):
      self.generator_train_output = generator.inference(generator.input_type,
        generator.noise)
      tf.summary.image('generator_generated_image', self.generator_train_output)
    with tf.variable_scope('generator_target'):
      self.generator_target_output = generator.inference(generator.input_type,
        generator.noise, trainable=False)
    self.generator = generator

    self.sync_generator_ops = []
    with tf.name_scope('generator_copy'):
      train_var = \
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'generator_train')
      target_var = \
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'generator_target')
      assert len(train_var) == len(target_var)
      assert len(train_var) > 0

      for i in range(len(train_var)):
        self.sync_generator_ops.append(tf.assign(target_var[i], train_var[i]))

    discriminator = Discriminator(self.num_classes, self.input_size)
    with tf.variable_scope('discriminator_train'):
      self.train_logits, self.discriminator_train_output = \
        discriminator.inference(discriminator.input_images)
      tf.summary.image('input_images', discriminator.input_images)
    with tf.variable_scope('discriminator_target'):
      self.target_logits, self.discriminator_target_output = \
        discriminator.inference(self.generator_train_output, trainable=False)
    self.discriminator = discriminator

    self.sync_discriminator_ops = []
    with tf.name_scope('discriminator_copy'):
      train_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        'discriminator_train')
      target_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        'discriminator_target')
      assert len(train_var) == len(target_var)
      assert len(train_var) > 0

      for i in range(len(train_var)):
        self.sync_discriminator_ops.append(tf.assign(target_var[i], train_var[i]))

    with tf.name_scope('generator_loss'):
      #  self.gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      #    logits=self.target_logits, labels=discriminator.labels))
      self.gloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=self.target_logits, labels=discriminator.labels))
      tf.summary.scalar('generator_loss', self.gloss)

    with tf.name_scope('discriminator_loss'):
      #  self.dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      #    logits=self.train_logits, labels=discriminator.labels))
      self.dloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=self.train_logits, labels=discriminator.labels))
      tf.summary.scalar('discriminator_loss', self.dloss)

    with tf.name_scope('optimization'):
      #  self.glearning_rate = tf.Variable(glearning_rate, trainable=False,
      #    name='generator_learning_rate')
      #  goptimizer = tf.train.GradientDescentOptimizer(self.glearning_rate)
      #  self.train_generator = goptimizer.minimize(self.gloss)

      #  self.dlearning_rate = tf.Variable(dlearning_rate, trainable=False,
      #    name='discriminator_learning_rate')
      #  doptimizer = tf.train.GradientDescentOptimizer(self.dlearning_rate)
      #  self.train_discriminator = doptimizer.minimize(self.dloss)
      self.loss = self.gloss + self.dloss
      self.learning_rate = tf.Variable(glearning_rate, trainable=False)
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_ops = optimizer.minimize(self.loss)

      #  tf.summary.scalar('generator_learning_rate', self.glearning_rate)
      #  tf.summary.scalar('discriminator_learning_rate', self.dlearning_rate)

    with tf.name_scope('evaluation'):
      prediction = tf.argmax(self.discriminator_train_output, axis=1)
      answer = tf.argmax(self.discriminator.labels, axis=1)
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, answer),
        tf.float32))
      tf.summary.scalar('accuracy', self.accuracy)

    with tf.name_scope('summary'):
      self.summary = tf.summary.merge_all()

  def prepare_folder(self):
    index = 0
    folder = 'gan-%d' % (index)
    while os.path.isdir(folder):
      index += 1
      folder = 'gan-%d' % (index)
    os.mkdir(folder)
    return folder

  def train(self, args):
    loader = PlantLoader(args.dbname)
    loader.load_data()

    # prepare data
    if args.load_all:
      training_data = loader.get_data()
      training_labels = loader.get_label()
    else:
      training_data = loader.get_training_data()
      training_labels = loader.get_training_labels()

    training_data = norm_images(training_data)
    training_labels = expand_label(training_labels)

    data_size = len(training_data)

    validation_data = norm_images(loader.get_validation_data())
    validation_labels = expand_label(loader.get_validation_labels())

    # prepare saver
    if args.saving:
      folder = self.prepare_folder()
      checkpoint = os.path.join(folder, 'gan')

      saver = tf.train.Saver()
      summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
        tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())

      noise = np.random.normal(0.0, 1.0,
        size=[args.batch_size, self.generator.noise_size])

      data_batch = training_data[0:args.batch_size, :]
      label_batch = training_labels[0:args.batch_size, :]

      for epoch in range(args.max_epoches + 1):
        offset = epoch % (data_size - args.batch_size)
        data_batch = training_data[offset:offset+args.batch_size, :]
        label_batch = training_labels[offset:offset+args.batch_size, :]

        # generate images
        sess.run(self.sync_generator_ops)
        generate_size = args.batch_size / 2
        noise = np.random.normal(0.0, 1.0,
          size=[generate_size, self.generator.noise_size])
        generated_images = sess.run(self.generator_target_output, feed_dict={
          self.generator.input_type: label_batch[:generate_size, 1:],
          self.generator.noise: noise,
        })
        #  print(np.std(generated_images), np.mean(generated_images))
        generated_labels = np.zeros(shape=[generate_size,
          self.discriminator.output_size])
        generated_labels[:, 0] = 1.0

        combined_data = np.concatenate([data_batch, generated_images], axis=0)
        combined_label = np.concatenate([label_batch, generated_labels], axis=0)

        if epoch % args.display_epoches == 0:
          gloss = sess.run(self.gloss, feed_dict={
            self.generator.noise: np.random.normal(0.0, 1.0,
              size=[args.batch_size, self.generator.noise_size]),
            self.generator.input_type: label_batch[:, 1:],
            self.discriminator.labels: label_batch,
            self.discriminator.keep_prob: 1.0,
          })

          dloss, accuracy = sess.run([self.dloss, self.accuracy], feed_dict={
            self.discriminator.input_images: data_batch,
            self.discriminator.labels: label_batch,
            self.discriminator.keep_prob: 1.0,
          })
          #  loss, gloss, dloss, accuracy = sess.run(
          #    [self.loss, self.gloss, self.dloss, self.accuracy],
          #    feed_dict={
          #      self.generator.noise: np.random.normal(0.0, 1.0,
          #        size=[args.batch_size, self.generator.noise_size]),
          #      self.generator.input_type: label_batch[:, 1:],
          #      self.discriminator.labels: label_batch,
          #      self.discriminator.input_images: data_batch,
          #      self.discriminator.keep_prob: 1.0,
          #    })

          valid_dloss, valid_accuracy = sess.run([self.dloss, self.accuracy],
            feed_dict={
              self.discriminator.input_images: validation_data,
              self.discriminator.labels: validation_labels,
              self.discriminator.keep_prob: 1.0,
            })

          logger.info('%d. gloss: %f, dloss: %f, classification: %f',
            epoch, gloss, dloss, accuracy)
          logger.info('validation dloss: %f, classification: %f',
            valid_dloss, valid_accuracy)


        #  sess.run(self.train_ops, feed_dict={
        #    self.generator.input_type: combined_label[:, 1:],
        #    self.generator.noise: np.random.normal(0.0, 1.0,
        #      [len(combined_label), self.generator.noise_size]),
        #    self.discriminator.input_images: combined_data,
        #    self.discriminator.labels: combined_label,
        #    self.discriminator.keep_prob: args.keep_prob,
        #  })
        # train generator
        if epoch % args.tau == 0:
          sess.run(self.sync_discriminator_ops)
          noise = np.random.normal(0.0, 1.0,
            size=[args.batch_size, self.generator.noise_size])
          sess.run(self.train_generator, feed_dict={
            self.generator.input_type: label_batch[:, 1:],
            self.generator.noise: noise,
            self.discriminator.keep_prob: 1.0,
            self.discriminator.labels: label_batch,
          })

        # train discriminator
        sess.run(self.train_discriminator, feed_dict={
          self.discriminator.input_images: combined_data,
          self.discriminator.labels: combined_label,
          self.discriminator.keep_prob: args.keep_prob,
        })

        if epoch % args.save_epoches == 0 and epoch != 0 and args.saving:
          saver.save(sess, checkpoint, global_step=epoch)

        if epoch % args.summary_epoches == 0 and epoch != 0 and args.saving:
          noise = np.random.normal(0.0, 1.0,
            size=[args.batch_size, self.generator.noise_size])
          summary = sess.run(self.summary, feed_dict={
            self.generator.input_type: label_batch[:, 1:],
            self.generator.noise: noise,
            self.discriminator.input_images: data_batch,
            self.discriminator.labels: label_batch,
            self.discriminator.keep_prob: 1.0,
          })
          summary_writer.add_summary(summary, global_step=epoch)


def main():
  parser = ArgumentParser()
  parser.add_argument('--dbname', dest='dbname', default='plants.sqlite3',
    type=str, help='dbname to load data for training')

  parser.add_argument('--glearning-rate', dest='glearning_rate',
    default=1e-2, type=float, help='generator learning rate')
  parser.add_argument('--dlearning-rate', dest='dlearning_rate',
    default=1e-3, type=float, help='discriminator learning rate')
  parser.add_argument('--tau', dest='tau',
    default=3, type=int, help='the frequency for updating generator')
  parser.add_argument('--epsilon', dest='epsilon',
    default=1, type=int, help='number of updates for generator')
  parser.add_argument('--max-epoches', dest='max_epoches', default=100000,
    type=int, help='max epoches to train model')
  parser.add_argument('--display-epoches', dest='display_epoches', default=10,
    type=int, help='epoches to evaluation')
  parser.add_argument('--save-epoches', dest='save_epoches', default=1000,
    type=int, help='epoches to save model')
  parser.add_argument('--summary-epoches', dest='summary_epoches', default=10,
    type=int, help='epoch to save summary')
  parser.add_argument('--batch-size', dest='batch_size', default=16,
    type=int, help='batch size to train model')
  parser.add_argument('--saving', dest='saving', default=False,
    type=bool, help='rather to save model or not')
  parser.add_argument('--keep-prob', dest='keep_prob', default=0.8,
    type=float, help='keep probability for dropout')
  parser.add_argument('--decay-epoch', dest='decay_epoch', default=10000,
    type=int, help='epoches to decay learning rate')

  parser.add_argument('--load-all', dest='load_all', default=False,
    type=bool, help='loading all training data')

  args = parser.parse_args()

  gan = PlantGAN(12, 32, args.glearning_rate, args.dlearning_rate)
  gan.train(args)



if __name__ == '__main__':
  main()

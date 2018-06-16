import tensorflow as tf
import numpy as np
import os
import sqlite3
import logging

from mnist_gan import GAN


logging.basicConfig()
logger = logging.getLogger('train_mnist_gan')
logger.setLevel(logging.INFO)

FLAGS = tf.app.flags.FLAGS
# hyper parameters
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate to train')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size to train')
tf.app.flags.DEFINE_integer('max_epoch', 10000, 'max epoches to train')
tf.app.flags.DEFINE_integer('decay_epoch', 10000,
  'epoches to decay learning rate')
tf.app.flags.DEFINE_float('keep_prob', 0.9, 'keep prob for dropout')

tf.app.flags.DEFINE_integer('display_epoch', 10, 'epoches to display result')
tf.app.flags.DEFINE_integer('summary_epoch', 10, 'epoches to save summary')
tf.app.flags.DEFINE_integer('save_epoch', 1000, 'epoches to save model')
tf.app.flags.DEFINE_string('db_path', './mnist.sqlite3', 'db to load')
tf.app.flags.DEFINE_bool('saving', False, 'saving model')


def load_data(db_path):
  if os.path.isfile(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM train;""")
    train_data = cursor.fetchall()

    cursor.execute("""SELECT * FROM test;""")
    test_data = cursor.fetchall()
    connection.close()
    return train_data, test_data


def parse_data(raw_data):
  data = []
  label = []
  for entry in raw_data:
    img = np.frombuffer(entry[0], dtype=np.uint8)
    data.append(np.reshape(img, [28, 28, 1]))
    label.append(entry[1])
  return np.array(data), np.array(label)


def prepare_folder():
  index = 0
  folder = 'mnist_gan_%d' % index
  while os.path.isdir(folder):
    index += 1
    folder = 'mnist_gan_%d' % index

  logger.info('creating folder %s...', folder)
  os.mkdir(folder)
  return folder


def main():
  logger.info('loading data from %s...', FLAGS.db_path)
  train_data, test_data = load_data(FLAGS.db_path)

  logger.info('parsing data...')
  train_images, train_labels = parse_data(train_data)
  test_images, test_labels = parse_data(test_data)

  train_data_size = len(train_images)

  logger.info('setting up model...')
  gan = GAN(FLAGS.learning_rate)

  if FLAGS.saving:
    logger.info('prepare saver...')
    saver = tf.train.Saver()

    folder = prepare_folder()
    checkpoint = os.path.join(folder, 'mnist_gan')
    summary_dir = os.path.join(folder, 'summary')

    logger.info('prepare summary writer...')
    summary_writer = tf.summary.FileWriter(summary_dir,
      tf.get_default_graph())

  logger.info('setting up session...')
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initializing variables...')
    sess.run(tf.global_variables_initializer())

    index = 0
    try:
      for epoch in range(FLAGS.max_epoch + 1):
        # get batch data
        start = index
        end = min(index + FLAGS.batch_size, train_data_size)

        index += FLAGS.batch_size
        if index >= train_data_size: index = 0

        batch_images = train_images[start:end, ...]
        batch_labels = train_labels[start:end, ...]

        if epoch % FLAGS.display_epoch == 0:
          losses = sess.run([gan.g_loss, gan.d_loss], feed_dict={
            gan.target_images: batch_images,
            gan.keep_prob: 1.0,
            gan.target_labels: batch_labels,
            gan.input_noise: np.random.randn(FLAGS.batch_size, 28, 28, 1),
          })
          logger.info('%d. loss: %s', epoch, str(losses))

        # training
        if epoch % 2 == 0:
          sess.run(gan.train_g, feed_dict={
            gan.target_images: batch_images,
            gan.keep_prob: FLAGS.keep_prob,
            gan.target_labels: batch_labels,
            gan.input_noise: np.random.randn(FLAGS.batch_size, 28, 28, 1),
          })

        sess.run(gan.train_d, feed_dict={
          gan.target_images: batch_images,
          gan.keep_prob: FLAGS.keep_prob,
          gan.target_labels: batch_labels,
          gan.input_noise: np.random.randn(FLAGS.batch_size, 28, 28, 1),
        })

        if FLAGS.saving and epoch % FLAGS.save_epoch == 0 and epoch != 0:
          logger.info('saving session...')
          saver.save(sess, checkpoint, global_step=epoch)

        if FLAGS.saving and epoch % FLAGS.summary_epoch == 0 and epoch != 0:
          summary_log = sess.run(gan.summary, feed_dict={
            gan.target_images: batch_images,
            gan.keep_prob: 1.0,
            gan.target_labels: batch_labels,
            gan.input_noise: np.random.randn(FLAGS.batch_size, 28, 28, 1),
          })
          summary_writer.add_summary(summary_log, global_step=epoch)
    except KeyboardInterrupt:
      logger.info('stopping...')

    if FLAGS.saving and epoch % FLAGS.save_epoch == 0 and epoch != 0:
      logger.info('saving last session...')
      saver.save(sess, checkpoint, global_step=epoch)


if __name__ == '__main__':
  main()

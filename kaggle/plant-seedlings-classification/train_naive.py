import tensorflow as tf
import numpy as np
import os
import signal
import logging
import sys

from naive import PlantNaiveModel
from plant_sampler import PlantSampler


logging.basicConfig()
logger = logging.getLogger('training')
logger.setLevel(logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('saving', False, 'saving model')
tf.app.flags.DEFINE_integer('display_epoches', 100,
  'display result per epoches')
tf.app.flags.DEFINE_integer('save_epoches', 10000,
  'save result per epoches')
tf.app.flags.DEFINE_integer('summary_epoches', 10,
  'save summary per epoches')
tf.app.flags.DEFINE_string('dbname', 'plants.sqlite3', 'prepared db to load')
tf.app.flags.DEFINE_bool('load_all', False, 'load all data for training')
tf.app.flags.DEFINE_integer('sample_worker_count', 100,
  'sampling worker count')

# hyper parameters
tf.app.flags.DEFINE_integer('input_size', 64, 'input image size')
tf.app.flags.DEFINE_integer('max_epoch', 300000, 'max epoch to train model')
tf.app.flags.DEFINE_integer('batch_size', 32, 'training batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
tf.app.flags.DEFINE_float('lambda_reg', 3e-3, 'regularization term')
tf.app.flags.DEFINE_float('keep_prob', 0.75, 'keep prob for dropout')


def train():
  sampler = PlantSampler(FLAGS.sample_worker_count,
    FLAGS.dbname, FLAGS.input_size, FLAGS.batch_size, FLAGS.load_all)
  sampler.start()

  valid_data, valid_label = sampler.get_validation_data()
  num_classes = valid_label.shape[1]

  model = PlantNaiveModel(FLAGS.input_size, num_classes,
    FLAGS.learning_rate, FLAGS.lambda_reg)

  if FLAGS.saving:
    index = 0
    folder = os.path.join('/tmp', 'plant_naive_%d' % (index))
    while os.path.isdir(folder):
      index += 1
      folder = os.path.join('/tmp', 'plant_naive_%d' % (index))

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
      tf.get_default_graph())
    checkpoint = os.path.join(folder, 'plant')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initializing variables...')
    sess.run(tf.global_variables_initializer())

    def stop(signal, frame):
      logger.info('stopping...')
      sampler.stop()
      sys.exit()

      if FLAGS.saving:
        saver.save(sess, checkpoint)
    signal.signal(signal.SIGINT, stop)

    logger.info('start training...')
    for epoch in range(FLAGS.max_epoch + 1):
      batch_data, batch_label = sampler.get_data()

      if epoch % FLAGS.display_epoches == 0:
        valid_index = np.random.permutation(
          np.arange(len(valid_data)))[:FLAGS.batch_size]
        valid_batch_data = valid_data[valid_index, ...]
        valid_batch_label = valid_label[valid_index, ...]

        loss, train, valid = sess.run(
          [model.loss, model.train_acc, model.valid_acc], feed_dict={
            model.images: batch_data,
            model.labels: batch_label,
            model.valid_images: valid_batch_data,
            model.valid_labels: valid_batch_label,
            model.keep_prob: 1.0,
          })

        logger.info('%d. loss: %f, train: %f, valid: %f',
          epoch, loss, train, valid)
        logger.info('queue size: %d', sampler.queue_size())

      if FLAGS.saving and epoch % FLAGS.save_epoches == 0 and epoch != 0:
        saver.save(sess, checkpoint, global_step=epoch)

      if FLAGS.saving and epoch % FLAGS.summary_epoches == 0:
        summary = sess.run(model.summary, feed_dict={
          model.images: batch_data,
          model.labels: batch_label,
          model.valid_images: valid_batch_data,
          model.valid_labels: valid_batch_label,
          model.keep_prob: 1.0,
        })
        summary_writer.add_summary(summary, global_step=epoch)

      sess.run(model.train_ops, feed_dict={
        model.images: batch_data,
        model.labels: batch_label,
        model.keep_prob: FLAGS.keep_prob,
      })

  stop()


if __name__ == '__main__':
  train()

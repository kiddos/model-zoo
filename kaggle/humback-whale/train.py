from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import sqlite3
from PIL import Image
import coloredlogs
import logging
import random

from humback_whale_model import HumbackWhaleModel
from humback_whale_data import HumbackWhaleData


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('saving', False, 'saving model')
tf.app.flags.DEFINE_string('data', 'humback-whale.sqlite3', 'sqlite3 data')

# hyperparameters
tf.app.flags.DEFINE_integer('image_width', 32, 'input image width')
tf.app.flags.DEFINE_integer('image_height', 32, 'input image height')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate to train')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.app.flags.DEFINE_integer('max_epoches', 100000, 'max epoch to train')
tf.app.flags.DEFINE_float('keep_prob', 0.8, 'keep prob for dropouts')

tf.app.flags.DEFINE_integer('display_epoches', 100, 'display epoches')
tf.app.flags.DEFINE_integer('save_epoches', 10000, 'save epoches')
tf.app.flags.DEFINE_integer('summary_epoches', 50, 'summary epoches')


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)


def train():
  data = HumbackWhaleData(FLAGS.data, FLAGS.image_width, FLAGS.image_height)
  num_classes = data.labels.max() + 1
  model = HumbackWhaleModel(FLAGS.image_width, FLAGS.image_height,
    num_classes, FLAGS.learning_rate)

  if FLAGS.saving:
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('/tmp/humback-whale/summary',
      tf.get_default_graph())

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initialize variables...')
    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.max_epoches + 1):
      images, labels = data.get_batch(FLAGS.batch_size)

      if epoch % FLAGS.display_epoches == 0:
        valid_images, valid_labels = data.get_batch(FLAGS.batch_size)

        loss, train_accuracy, valid_accuracy = sess.run([model.loss,
          model.train_accuarcy, model.valid_accuarcy],
          feed_dict={
            model.input_images: images,
            model.label: labels,
            model.valid_images: valid_images,
            model.valid_label: valid_labels,
            model.keep_prob: 1.0,
          })
        logger.info('%d. loss: %f | train: %f | valid: %f',
          epoch, loss, train_accuracy, valid_accuracy)

      # train
      sess.run(model.train_ops, feed_dict={
        model.input_images: images,
        model.label: labels,
        model.keep_prob: FLAGS.keep_prob,
      })

      if FLAGS.saving and epoch % FLAGS.save_epoches == 0 and epoch != 0:
        saver.save(sess, '/tmp/humback-whale/model', global_step=epoch)

      if FLAGS.saving and epoch % FLAGS.summary_epoches == 0:
        valid_images, valid_labels = data.get_batch(FLAGS.batch_size)

        summary = sess.run(model.summary, feed_dict={
          model.input_images: images,
          model.label: labels,
          model.valid_images: valid_images,
          model.valid_label: valid_labels,
          model.keep_prob: 1.0,
        })
        summary_writer.add_summary(summary, global_step=epoch)


def main(_):
  train()


if __name__ == '__main__':
  tf.app.run()

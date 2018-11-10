import tensorflow as tf
import numpy as np
import sys
import coloredlogs
import logging
import os
from argparse import ArgumentParser

from cifar10_data_util import CIFAR10Data


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('cifar10')
logger.setLevel(logging.INFO)


def train(args):
  logger.info('setting up models...')
  model = CIFAR10Model(args.inference, learning_rate=args.learning_rate)

  logger.info('preparing cifar10 data...')
  cifar10_data = CIFAR10Data(args.dbname)
  training_data, training_label = cifar10_data.get_training_data()
  valid_data, valid_label = cifar10_data.get_validation_data()
  training_data = np.concatenate([training_data, valid_data], axis=0)
  training_label = np.concatenate([training_label, valid_label], axis=0)

  valid_data, valid_label = cifar10_data.get_test_data()

  logger.info('training data: %s', str(training_data.shape))
  logger.info('validation data: %s', str(valid_data.shape))

  saving = (args.saving == 'True')

  if saving:
    folder = model.prepare_folder()
    checkpoint = os.path.join(folder, 'cifar10')
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(folder, 'summary'),
      tf.get_default_graph())

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initializing variables...')
    sess.run(tf.global_variables_initializer())

    training_size = len(training_data)
    valid_size = len(valid_data)
    valid_index = 0
    offset = 0
    for epoch in range(args.max_epoches + 1):
      training_data_batch = training_data[offset:offset+args.batch_size, :]
      training_label_batch = training_label[offset:offset+args.batch_size, :]

      offset += args.batch_size
      if offset >= training_size - args.batch_size and offset < training_size:
        offset = training_size - args.batch_size
      elif offset >= training_size:
        offset = 0

      if epoch % args.display_epoches == 0:
        to = valid_index + args.batch_size
        valid_data_batch = valid_data[valid_index:to, :]
        valid_label_batch = valid_label[valid_index:to, :]

        loss, train, valid = sess.run(
          [model.loss, model.train_accuracy, model.validation_accuracy],
          feed_dict={
            model.input_images: training_data_batch,
            model.labels: training_label_batch,
            model.validation_images: valid_data_batch,
            model.validation_labels: valid_label_batch,
            model.keep_prob: 1.0
          })

        valid_index = to % (valid_size - args.batch_size)

        logger.info('%d. loss: %f, train: %f, valid: %f',
          epoch, loss, train, valid)

      # train the model
      sess.run(model.train_ops, feed_dict={
        model.input_images: training_data_batch,
        model.labels: training_label_batch,
        model.keep_prob: args.keep_prob,
      })

      if epoch % args.save_epoches == 0 and epoch != 0 and saving:
        saver.save(sess, checkpoint, global_step=epoch)

      if epoch % args.summary_epoches == 0 and epoch != 0 and saving:
        summary = sess.run(model.summary, feed_dict={
          model.input_images: training_data_batch,
          model.labels: training_label_batch,
          model.validation_images: valid_data_batch,
          model.validation_labels: valid_label_batch,
          model.keep_prob: 1.0
        })
        summary_writer.add_summary(summary, global_step=epoch)

      if epoch % args.decay_epoches == 0 and epoch != 0:
        sess.run(model.decay_lr)


def main():
  parser = ArgumentParser()

  parser.add_argument('--dbname', dest='dbname', default='cifar.sqlite3',
    help='dbname to load for training')

  parser.add_argument('--inference', dest='inference',
    default='inference_v0', help='inference function to use')

  parser.add_argument('--learning-rate', dest='learning_rate', type=float,
    default=1e-3, help='learning rate for training')
  parser.add_argument('--batch-size', dest='batch_size', type=int,
    default=64, help='batch size for training')
  parser.add_argument('--max-epoches', dest='max_epoches', type=int,
    default=300000, help='max epoches to train')
  parser.add_argument('--display-epoches', dest='display_epoches', type=int,
    default=10, help='epoches to display training result')
  parser.add_argument('--save-epoches', dest='save_epoches', type=int,
    default=10000, help='epoches to save training result')
  parser.add_argument('--summary-epoches', dest='summary_epoches', type=int,
    default=10, help='epoches to save training summary')
  parser.add_argument('--decay-epoches', dest='decay_epoches', type=int,
    default=20000, help='epoches to decay learning rate for training')
  parser.add_argument('--keep-prob', dest='keep_prob', type=float,
    default=0.8, help='keep probability for dropout')
  parser.add_argument('--saving', dest='saving', type=str,
    default='False', help='rather to save the training result')
  args = parser.parse_args()

  train(args)


if __name__ == '__main__':
  main()

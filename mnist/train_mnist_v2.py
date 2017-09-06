import tensorflow as tf
import numpy as np
import logging
import os
from argparse import ArgumentParser

from mnist_prepare import load
from mnist_v2 import MNISTV2


logging.basicConfig()
logger = logging.getLogger('train mnist v2')
logger.setLevel(logging.INFO)


def train():
  parser = ArgumentParser()
  parser.add_argument('--learning-rate', dest='learning_rate',
    default=1e-3, type=float, help='learning rate')
  parser.add_argument('--keep-prob', dest='keep_prob',
    default=0.8, type=float, help='keep probability')
  parser.add_argument('--db', dest='db', default='mnist.sqlite3',
    help='input sqlite3 db')
  parser.add_argument('--max-epoch', dest='max_epoch',
    default=100000, help='max epoch to train')
  parser.add_argument('--batch-size', dest='batch_size',
    default=256, type=int, help='batch size for training')
  parser.add_argument('--save-epoch', dest='save_epoch',
    default=1000, type=int, help='epoch for saving')
  parser.add_argument('--display-epoch', dest='display_epoch',
    default=100, type=int, help='display epoch')
  parser.add_argument('--model-path', dest='model_path',
    default='mnist_v2', help='model path to save')
  parser.add_argument('--decay-epoch', dest='decay_epoch',
    default=10000, type=int, help='decay epoch')
  args = parser.parse_args()

  logger.info('creating model directory')
  index = 0
  model_path = '%s_%d' % (args.model_path, index)
  while os.path.isdir(model_path):
    index += 1
    model_path = '%s_%d' % (args.model_path, index)
  os.mkdir(model_path)
  checkpoint_path = os.path.join(model_path, args.model_path)

  train_images, train_label, test_images = load(args.db)
  model = MNISTV2(learning_rate=args.learning_rate)

  saver = tf.train.Saver()

  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    indices = np.arange(len(train_images))
    for epoch in range(args.max_epoch):
      i = np.random.permutation(indices)[:args.batch_size]
      image_batch = train_images[i, :]
      label_batch = train_label[i, :]

      if epoch % args.display_epoch == 0:
        loss, accuracy = sess.run([model.loss, model.accuracy],
          feed_dict={
          model.images: image_batch,
          model.labels: label_batch,
          model.keep_prob: 1.0
        })
        logger.info('%d. loss: %f, accuracy: %f' %
          (epoch, loss, accuracy))
      if epoch % args.save_epoch == 0 and epoch != 0:
        logger.info('saving model...')
        saver.save(sess, checkpoint_path, global_step=epoch)

      sess.run(model.train_ops, feed_dict={
        model.images: image_batch,
        model.labels: label_batch,
        model.keep_prob: args.keep_prob
      })

      if epoch % args.decay_epoch == 0 and epoch != 0:
        logger.info('decay learning rate...')
        sess.run(model.decay_lr)


if __name__ == '__main__':
  train()

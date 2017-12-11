import tensorflow as tf
import numpy as np
import os
import logging
from argparse import ArgumentParser

from yolo_face_v7 import YOLOFace
from yolo_face_v7_batcher import ImageBatch


logging.basicConfig()
logger = logging.getLogger('train yolo face')
logger.setLevel(logging.INFO)


def train():
  parser = ArgumentParser()
  parser.add_argument('--lambda-coord', dest='lambda_coord',
    default=5.0, type=float, help='coordinate loss adjustment')
  parser.add_argument('--lambda-noobj', dest='lambda_noobj',
    default=0.5, type=float, help='no object loss adjustment')

  parser.add_argument('--learning-rate', dest='learning_rate',
    default=1e-5, type=float, help='learning rate to train')
  parser.add_argument('--batch-size', dest='batch_size',
    default=10, type=int, help='batch size to train model')
  parser.add_argument('--max-epoch', dest='max_epoch',
    default=200000, type=int, help='max epoch train model')
  parser.add_argument('--keep-prob', dest='keep_prob',
    default=0.8, type=float, help='keep prob for dropout')
  parser.add_argument('--display-epoch', dest='display_epoch',
    default=10, type=int, help='epochs to display output')
  parser.add_argument('--save-epoch', dest='save_epoch',
    default=1000, type=int, help='epochs to save')
  parser.add_argument('--decay-epoch', dest='decay_epoch',
    default=50000, type=int, help='epoch to decay learning rate')
  parser.add_argument('--save', dest='save',
    default=False, type=bool, help='saving model')

  parser.add_argument('--image-dir', dest='image_dir',
    default='WIDER_train/images', help='image directory')
  parser.add_argument('--label-txt', dest='label_txt',
    default='wider_face_split/wider_face_train_bbx_gt.txt', help='label txt')
  args = parser.parse_args()

  logger.info('setting up batcher...')
  batcher = ImageBatch(args.image_dir, args.label_txt)

  logger.info('setting up model...')
  model = YOLOFace(learning_rate=args.learning_rate,
    lambda_coord=args.lambda_coord,
    lambda_noobj=args.lambda_noobj)

  if args.save:
    logger.info('setting up saver...')
    saver = tf.train.Saver()
    num_model = 0
    checkpoint_path = 'yolo_face_v7_%d' % (num_model)
    while os.path.isdir(checkpoint_path):
      num_model += 1
      checkpoint_path = 'yolo_face_v7_%d' % (num_model)
    logger.info('creating directory %s...' % (checkpoint_path))
    os.mkdir(checkpoint_path)

    summary_writer = tf.summary.FileWriter(
      os.path.join(checkpoint_path, 'summary'),
      graph=tf.get_default_graph(),
      flush_secs=60)
    checkpoint_path = os.path.join(checkpoint_path, 'yolo_face')

  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger.info('initializing variables...')
    sess.run(tf.global_variables_initializer())

    for epoch in range(args.max_epoch + 1):
      image_batch, indicator_batch, coord_batch, size_batch = \
          batcher.next_batch(args.batch_size)
      feed = {
        model.images: image_batch,
        model.indicator_label: indicator_batch,
        model.coordinate_label: coord_batch,
        model.size_label: size_batch,
        model.keep_prob: 1.0
      }

      if epoch % args.display_epoch == 0:
        loss, indicator_loss, coord_loss, size_loss, noobj_loss, summary = \
          sess.run([model.loss, model.indicator_loss, model.coord_loss,
            model.size_loss, model.noobj_loss, model.summary],
            feed_dict=feed)
        logger.info('%d. loss: (%f, %f, %f, %f, %f)' % (
          epoch, loss, indicator_loss, coord_loss, size_loss, noobj_loss))
        if args.save:
          summary_writer.add_summary(summary, global_step=epoch)
      if args.save and epoch % args.save_epoch == 0 and epoch != 0:
        logger.info('saving model...')
        saver.save(sess, checkpoint_path, global_step=epoch)
      if epoch % args.decay_epoch == 0 and epoch != 0:
        sess.run(model.decay_lr)

      feed[model.keep_prob] = args.keep_prob
      # train ops
      sess.run(model.train_ops, feed_dict=feed)


if __name__ == '__main__':
  train()

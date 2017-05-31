import tensorflow as tf
import os
from argparse import ArgumentParser

from kinect_pose import KinectPoseModel
from kinect_pose_prepare import FreiburgData


def main():
  parser = ArgumentParser()
  parser.add_argument('--db', dest='db', default='freiburg1_xyz.sqlite3',
    help='input database')
  parser.add_argument('--batch-size', type=int, dest='batch_size',
    default=64, help='batch size for training')
  parser.add_argument('--max-epoch', type=int, dest='max_epoch',
    default=100000, help='max epoch for training')
  parser.add_argument('--output-epoch', type=int, dest='output_epoch',
    default=10, help='epoch for display and saving')
  parser.add_argument('--keep-prob', type=float, dest='keep_prob',
    default=0.8, help='keep probability for drop out')
  parser.add_argument('--decay-epoch', type=int, dest='decay_epoch',
    default=5000, help='decay learning rate epoch')

  args = parser.parse_args()
  data_loader = FreiburgData(args.db)

  input_width = 640
  input_height = 480
  input_channel = 1
  output_size = 7

  model = KinectPoseModel(input_width, input_height, input_channel,
    output_size, model_name='KinectPose', saving=True)

  with tf.Session() as sess:
    model.train_with_loader(sess, data_loader,
      batch_size=args.batch_size,
      output_period=args.output_epoch,
      decay_epoch=args.decay_epoch,
      keep_prob=args.keep_prob,
      max_epoch=args.max_epoch)


if __name__ == '__main__':
  main()

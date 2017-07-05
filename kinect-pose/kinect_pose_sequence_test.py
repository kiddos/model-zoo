import os
import logging
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

from kinect_pose import KinectPoseModel
from kinect_pose_prepare import FreiburgData


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_seq(seq_images, checkpoint, init_pose):
  input_width, input_height, input_channel = 640, 480, 1
  output_size = 7
  model = KinectPoseModel(input_width, input_height, input_channel,
    output_size, model_name='KinectPose', saving=False)

  poses = [init_pose]
  if os.path.isfile(checkpoint + '.index') and \
      os.path.isfile(checkpoint + '.meta'):
    with tf.Session() as sess:
      model.load(sess, checkpoint)

      for i in range(len(seq_images) - 1):
        p = np.expand_dims(seq_images[i, :] / 5000.0, axis=0)
        n = np.expand_dims(seq_images[i + 1, :] / 5000.0, axis=0)
        prediction = np.squeeze(model.predict(sess, p, n), axis=0)
        poses.append(poses[-1] + prediction)
  return np.array(poses)


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-db', dest='input_db',
    default='freiburg1_xyz.sqlite3',
    help='input dataset')
  parser.add_argument('--checkpoint', dest='checkpoint',
    default='KinectPose2/KinectPose-100000',
    help='checkpoint path')
  args = parser.parse_args()

  data = FreiburgData(args.input_db)
  logger.info(data.depth_images.shape)

  predicted_poses = predict_seq(data.depth_images, args.checkpoint,
    data.depth_labels[0, :])

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(data.depth_labels[:, 0], data.depth_labels[:, 1],
    data.depth_labels[:, 2], 'g')
  ax.plot(predicted_poses[:, 0], predicted_poses[:, 1],
    predicted_poses[:, 2], 'r')
  ax.set_xlabel('$X$')
  ax.set_ylabel('$Y$')
  plt.show()


if __name__ == '__main__':
  main()

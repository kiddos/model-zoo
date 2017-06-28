import tensorflow as tf
import numpy as np
import os
from argparse import ArgumentParser

from kinect_pose import KinectPoseModel
from kinect_pose_prepare import FreiburgData


def main():
  parser = ArgumentParser()
  parser.add_argument('-i', dest='input_data',
    default='freiburg1_xyz.sqlite3', help='input data set')
  parser.add_argument('-t', dest='test_size',
    default=10000, type=int, help='test data size')
  parser.add_argument('-b', dest='batch_size',
    default=100, type=int, help='batch size for testing')
  parser.add_argument('-c', dest='checkpoint',
    default='KinectPose2/KinectPose-100000')
  args = parser.parse_args()

  input_width, input_height, input_channel = 640, 480, 1
  output_size = 7
  if os.path.isfile(args.checkpoint + '.index') and \
      os.path.isfile(args.checkpoint + '.meta'):
    print('preparing data loader...')
    data_loader = FreiburgData(args.input_data)

    print('loading pre-train model...')
    model = KinectPoseModel(input_width, input_height, input_channel,
      output_size, model_name='KinectPose', saving=False)
    with tf.Session() as sess:
      model.load(sess, args.checkpoint)

      batches = args.test_size / args.batch_size
      outputs = np.zeros(shape=[output_size], dtype=np.float32)
      for i in range(batches):
        p, n, l = data_loader.diff_depth_batch(args.batch_size)
        prediction = model.predict(sess, p, n)
        outputs += np.sum(np.abs(prediction - l), axis=0)
        print('processing batch %d...' % (i))
      outputs /= args.test_size
      print(outputs)
  else:
    print('checkpoint not exists.')


if __name__ == '__main__':
  main()

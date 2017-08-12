from __future__ import print_function

import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import os
import cv2

from yolor_model import YOLORotationModel


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-width', dest='input_width',
    default=160, type=int, help='input image width')
  parser.add_argument('--input-height', dest='input_height',
    default=120, type=int, help='input image height')
  parser.add_argument('--output-width', dest='output_width',
    default=16, type=int, help='output image width')
  parser.add_argument('--output-height', dest='output_height',
    default=12, type=int, help='output image height')
  parser.add_argument('--output-class', dest='output_classes',
    default=10, type=int, help='number of output class')
  parser.add_argument('--checkpoint', dest='checkpoint',
    default='', type=str, help='checkpoint to load')
  args = parser.parse_args()

  model = YOLORotationModel(
    args.input_width, args.input_height, 3,
    args.output_width, args.output_height, 1, args.output_classes,
    model_name='YOLOWithRotation', saving=True)

  config = tf.ConfigProto()
  with tf.Session(config=config) as sess:
    if args.checkpoint != '' and \
        os.path.isfile(args.checkpoint + '.meta') and \
        os.path.isfile(args.checkpoint + '.index'):
      print('loading checkpoint %s...' % (args.checkpoint))
      model.load(sess, args.checkpoint)

      camera = cv2.VideoCapture()
      camera.open(1)
      if camera.isOpened():
        while True:
          _, img = camera.read()
          print(img.shape, (args.input_width, args.input_height))
          inputs = cv2.resize(img, (args.input_width, args.input_height))
          inputs = np.expand_dims(inputs, 0)
          cv2.imshow('Camera image', img)
          key = cv2.waitKey(10)
          if key in [ord('q'), ord('Q'), 10, 27]:
            break
    else:
      print('no trained model found in %s' % (args.checkpoint))


if __name__ == '__main__':
  main()

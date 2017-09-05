from __future__ import print_function

import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import os
import cv2
import sys

from yolo_model import YOLOModel
from rotation_invariance_model import RotationalInvarianceModel


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
  parser.add_argument('--yolo-checkpoint', dest='yolo_checkpoint',
    default='YOLO1/YOLO-200000',
    type=str, help='checkpoint to load')
  parser.add_argument('--rotation-checkpoint', dest='rotation_checkpoint',
    default='RotationalInvarianceModel6/RotationalInvarianceModel-100000',
    type=str, help='checkpoint to load')
  parser.add_argument('--threshold', dest='threshold',
    default=0.5, type=float, help='threshold for output')
  args = parser.parse_args()

  # yolo graph
  yolo_graph = tf.Graph()
  with yolo_graph.as_default() as g:
    yolo_model = YOLOModel(args.input_width, args.input_height, 3,
      args.output_width, args.output_height, 1, saving=False)
  yolo_sess = tf.Session(graph=g)
  # rotation graph
  rotation_graph = tf.Graph()
  with rotation_graph.as_default() as g:
    rotation_model = RotationalInvarianceModel(64, 3, 10,
      model_name='RotationalInvarianceModel',
      saving=False)
  rotation_sess = tf.Session(graph=g)

  if args.yolo_checkpoint != '' and \
      os.path.isfile(args.yolo_checkpoint + '.meta') and \
      os.path.isfile(args.yolo_checkpoint + '.index') and \
      args.rotation_checkpoint != '' and \
      os.path.isfile(args.rotation_checkpoint + '.meta') and \
      os.path.isfile(args.rotation_checkpoint + '.index'):
    yolo_model.load(yolo_sess, args.yolo_checkpoint)

    camera = cv2.VideoCapture(0)
    if camera.isOpened():
      while True:
        _, img = camera.read()
        inputs = cv2.resize(img, (args.input_width, args.input_height))
        inputs = np.expand_dims(inputs, 0)
        xy_output, size_output, indicator = \
            yolo_model.predict(yolo_sess, inputs)
        roi_display = np.copy(img)

        valid = (indicator > args.threshold).squeeze(-1)
        scores = np.sort(indicator[valid, :]).squeeze(-1)[::-1]
        xys = xy_output[valid, :]
        sizes = size_output[valid, :]
        print(scores)
        print('\rfound: %d | max score: %f' % (len(xys), indicator.max()))
        sys.stdout.flush()
        for i in range(len(xys)):
          if indicator[valid, :][i] == scores[0]:
            x = xys[i, 0] * 640
            y = xys[i, 1] * 480
            w = sizes[i, 0] * 640
            h = sizes[i, 1] * 480
            p1 = (int(x - w / 2), int(y - h / 2))
            p2 = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(roi_display, p1, p2, (0, 255, 0), 2)
            rotation_model.predict(rotation_sess,
                inputs[p1[1]:p2[1], p1[0]:p2[0], :].expand_dim(0))

        cv2.imshow('Camera image', img)
        cv2.imshow('Indicator',
          cv2.resize(np.squeeze(indicator, axis=0), (640, 480)))
        cv2.imshow('ROI', roi_display)
        key = cv2.waitKey(10)
        if key in [ord('q'), ord('Q'), 10, 27]:
          break
  yolo_sess.close()
  rotation_sess.close()


if __name__ == '__main__':
  main()

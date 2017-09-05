import tensorflow as tf
import numpy as np
import os
import logging
import sys
import cv2
from argparse import ArgumentParser

from roi import ROIModel
from image_client import ImageClient


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ROIInference(object):
  def __init__(self, input_width, input_height, input_channel,
      output_width, output_height, checkpoint):
    logger.info('initializing roi model...')
    self.model = ROIModel(input_width, input_height, input_channel,
        output_width, output_height, saving=False)
    logger.info('create tensorflow session')
    config = tf.ConfigProto()
    self.sess = tf.Session(config=config)
    if os.path.isfile(checkpoint + '.index') and \
        os.path.isfile(checkpoint + '.meta'):
      logger.info('loading checkpoint...')
      self.model.load(self.sess, checkpoint)
    else:
      logger.info('fail to load checkpoint %s' % (checkpoint))
      sys.exit(-1)
    self.input_width = input_width
    self.input_height = input_height
    self.input_channel = input_channel

  def inference(self, image):
    resized = cv2.resize(image, (self.input_width, self.input_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    grid = self.model.predict(self.sess, np.expand_dims(rgb, axis=0))
    return grid


  def __del__(self):
    logger.info('closing tensorflow session')
    self.sess.close()


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-width', dest='input_width',
    default=200, help='model input width')
  parser.add_argument('--input-height', dest='input_height',
    default=150, help='model input height')
  parser.add_argument('--input-channel', dest='input_channel',
    default=3, help='model input channel')
  parser.add_argument('--output-width', dest='output_width',
    default=16, help='output grid width')
  parser.add_argument('--output-height', dest='output_height',
    default=12, help='output grid height')

  parser.add_argument('--ip', dest='ip', default='localhost',
    type=str, help='server ip')
  parser.add_argument('--port', dest='port', default=50051,
    type=int, help='server port')
  parser.add_argument('--camera-id', dest='camera_id', default=0,
    type=int, help='local camera id')
  parser.add_argument('--mode', dest='mode', default='camera',
    type=str, help='mode to open (eg. camera/grpc)')

  parser.add_argument('--threshold', dest='threshold',
    default=0.2, type=float, help='threshold to be consider as object')
  parser.add_argument('--checkpoint', dest='checkpoint',
    default='ROIModel0/ROIModel-60000', help='checkpoint path')
  args = parser.parse_args()

  if args.mode == 'camera':
    host = args.camera_id
  else:
    host = '%s:%d' % (args.ip, args.port)
  client = ImageClient(host)
  inference = ROIInference(args.input_width, args.input_height,
    args.input_channel, args.output_width, args.output_height,
    args.checkpoint)

  while True:
    image = client.get_image()
    cv2.imshow('Client Image', image)

    grid = inference.inference(image)
    indicator = grid[0, :, :, 0] > args.threshold
    grid_display = indicator * 255.0
    grid_display[grid_display > 255] = 255
    grid_display = grid_display.astype(np.uint8)
    cv2.imshow('Indicator',
      cv2.resize(grid_display, (image.shape[1], image.shape[0])))

    roi_display = np.copy(image)
    roi = grid[0, indicator, :]
    for i in range(len(roi)):
      cx = roi[i, 1] * image.shape[1]
      cy = roi[i, 2] * image.shape[0]
      w = roi[i, 3] * image.shape[1]
      h = roi[i, 4] * image.shape[0]
      x1 = np.max([int(cx - w / 2), 0])
      x2 = np.min([int(cx + w / 2), image.shape[1]])
      y1 = np.max([int(cy - h / 2), 0])
      y2 = np.min([int(cy + h / 2), image.shape[0]])
      p1 = (x1, y1)
      p2 = (x2, y2)
      cv2.rectangle(roi_display, p1, p2, (0, 255, 0), 2)
    cv2.imshow('ROI', roi_display)

    key = cv2.waitKey(20)
    if key in [ord('q'), ord('Q')]:
      break


if __name__ == '__main__':
  main()

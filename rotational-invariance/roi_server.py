import tensorflow as tf
import numpy as np
import grpc
import logging
import time
from concurrent import futures
from argparse import ArgumentParser

from data_util.tiny_yolo_util import NUM_OUTPUT_CLASSES
from tiny_yolo_v2 import TinyYOLOV2
from rotational_invariance import ROI, Rect
from rotational_invariance_pb2_grpc import ROIServiceServicer
from rotational_invariance_pb2_grpc import add_ROIServiceServicer_to_server


class ROIServicer(ROIServiceServicer):
  def __init__(self, image_width, image_height, image_channel,
      checkpoint, threshold):
    logging.basicConfig()
    self.logger = logging.getLogger('roi server')
    self.logger.setLevel(logging.INFO)
    self.logger.info('starting tensorflow session')
    self.sess = tf.Session()

    self.logger.info('initializing tiny yolo model')
    self.model = TinyYOLOV2(image_width, image_height, image_channel)
    self.model.load(self.sess, checkpoint)
    self.threshold = threshold

    self.image_width = image_width
    self.image_height = image_height

  def GetROI(self, image, context):
    img = np.frombuffer(image.data, np.uint8).reshape(
      [1, image.height, image.width, 1])
    grid = self.model.predict(self.sess, img)

    valid = grid[:, :, NUM_OUTPUT_CLASSES]
    rois = grid[valid, NUM_OUTPUT_CLASSES:]
    roi = ROI()
    rects = []
    for i in range(len(rois)):
      rect = Rect()
      rect.w = rois[i, 2] * self.image_width
      rect.h = rois[i, 3] * self.image_height
      rect.x = rois[i, 0] * self.image_width - rect.w / 2
      rect.y = rois[i, 1] * self.image_height - rect.h / 2
      rects.append(rect)
    roi.num_roi = len(rects)
    roi.rects = rects
    return roi

  def __del__(self):
    self.logger.info('closing session')
    self.sess.close()


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-width', dest='input_width', default=240,
    type=int, help='input image width')
  parser.add_argument('--input-height', dest='input_height', default=180,
    type=int, help='input image height')
  parser.add_argument('--input-channel', dest='input_channel', default=1,
    type=int, help='input image channel')
  parser.add_argument('--checkpoint', dest='checkpoint',
    default='TinyYOLOV20/TinyYOLOV2-100000',
    type=int, help='input image channel')

  parser.add_argument('--ip', dest='ip', default='0.0.0.0',
    type=str, help='server ip')
  parser.add_argument('--port', dest='port', default=50051,
    type=int, help='server port')
  args = parser.parse_args()

  servicer = ROIServicer(args.input_width, args.input_height,
    args.input_channel)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  add_ROIServiceServicer_to_server(servicer, server)
  server.add_insecure_port('%s:%d' % (args.ip, args.port))
  server.start()

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  main()

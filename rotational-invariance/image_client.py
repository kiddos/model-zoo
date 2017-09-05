import grpc
import time
import cv2
import logging
import sys
import numpy as np
from argparse import ArgumentParser

from rotational_invariance_pb2 import Null
from rotational_invariance_pb2_grpc import ImageServiceStub


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageClient(object):
  def __init__(self, host):
    self.host = host
    self.mode = 'camera'
    if isinstance(self.host, str):
      logger.info('starting client, connecting to %s' % (host))
      self.channel = grpc.insecure_channel('%s' % (host))
      self.stub = ImageServiceStub(self.channel)
      self.mode = 'grpc'
    elif isinstance(self.host, int):
      logger.info('starting local camera %d...' % (self.host))
      self.camera = cv2.VideoCapture(self.host)
      if not self.camera.isOpened():
        logger.info('unable to open local camera %d' % (self.host))
        sys.exit(-1)

  def get_image(self):
    """
    retrieve image from different source
    both outputs a BGR image
      grpc: from image server
      camera: from local camera
    """
    if self.mode == 'grpc':
      image = self.stub.GetImage(Null())
      img = np.fromstring(image.data,
        dtype=np.uint8).reshape([image.height, image.width, image.channel])
      return img
    elif self.mode == 'camera':
      _, image = self.camera.read()
      return image
    return None

  def run(self):
    try:
      while True:
        img = self.get_image()
        cv2.imshow('Server Image', img)
        key = cv2.waitKey(20)
        if key in [ord('q'), ord('Q')]:
          break
    except KeyboardInterrupt:
      logger.info('close client')


def main():
  parser = ArgumentParser()
  parser.add_argument('--ip', dest='ip', default='localhost',
    type=str, help='server ip')
  parser.add_argument('--port', dest='port', default=50051,
    type=int, help='server port')
  parser.add_argument('--camera-id', dest='camera_id', default=0,
    type=int, help='local camera id')
  parser.add_argument('--mode', dest='mode', default='camera',
    type=str, help='mode to open (eg. camera/grpc)')
  args = parser.parse_args()

  if args.mode == 'camera':
    host = args.camera_id
  else:
    host = '%s:%d' % (args.ip, args.port)
  if host is not None:
    client = ImageClient(host)
    client.run()


if __name__ == '__main__':
  main()

import grpc
import cv2
import sys
import logging
import time
from concurrent import futures
from argparse import ArgumentParser

from rotational_invariance_pb2 import Image, Null
from rotational_invariance_pb2_grpc import ImageServiceServicer
from rotational_invariance_pb2_grpc import add_ImageServiceServicer_to_server


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageServicer(ImageServiceServicer):
  def __init__(self, camera_id, output_size):
    self.output_size = output_size
    self.camera = cv2.VideoCapture(camera_id)
    if not self.camera.isOpened():
      logger.info('camera %d cannot be opened' % (camera_id))
      sys.exit(0)
    else:
      logger.info('camera %d opened' % (camera_id))

  def GetImage(self, request, context):
    img = Image()
    _, frame = self.camera.read()
    output_image = cv2.resize(frame, self.output_size)
    img.width = output_image.shape[1]
    img.height = output_image.shape[0]
    img.channel = output_image.shape[2]
    img.data = output_image.tobytes()
    return img


def main():
  parser = ArgumentParser()
  parser.add_argument('--output-width', dest='output_width', default=240,
    type=int, help='output stream width')
  parser.add_argument('--output-height', dest='output_height', default=180,
    type=int, help='output stream height')
  parser.add_argument('--camera-id', dest='camera_id', default=0,
    type=int, help='camera id to open')

  parser.add_argument('--ip', dest='ip', default='0.0.0.0',
    type=str, help='server ip')
  parser.add_argument('--port', dest='port', default=50051,
    type=int, help='server port')
  args = parser.parse_args()

  servicer = ImageServicer(args.camera_id,
    (args.output_width, args.output_height))
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  add_ImageServiceServicer_to_server(servicer, server)
  server.add_insecure_port('%s:%d' % (args.ip, args.port))
  server.start()

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    logger.info('stopping server...')
    server.stop(0)


if __name__ == '__main__':
  main()

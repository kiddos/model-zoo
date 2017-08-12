from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import logging
import cv2
from argparse import ArgumentParser

from data_util.tiny_yolo_util import NUM_BOUNDING_BOX, NUM_OUTPUT_CLASSES
from data_util.tiny_yolo_util import OUTPUT_CHARACTERS
from tiny_yolo_v2 import TinyYOLOV2


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
  parser = ArgumentParser()
  parser.add_argument('--checkpoint', dest='checkpoint',
    default='TinyYOLOV20/TinyYOLOV2-100000')
  parser.add_argument('--input-width', dest='input_width', default=240,
    type=int, help='input image width')
  parser.add_argument('--input-height', dest='input_height', default=180,
    type=int, help='input image height')
  parser.add_argument('--input-channel', dest='input_channel', default=1,
    type=int, help='input image channel')
  parser.add_argument('--threshold', dest='threshold', default=0.6,
    type=float, help='threshold for inference')
  parser.add_argument('--output-dir', dest='output_dir',
    default='saved_images', help='image saving directory')
  parser.add_argument('--num-camera', dest='num_camera',
    default=2, type=int, help='image saving directory')
  args = parser.parse_args()

  model = TinyYOLOV2(args.input_width, args.input_height, args.input_channel,
    saving=False)

  saved_index = 0
  if not os.path.isdir(args.output_dir):
    logger.info('make output directory...')
    os.mkdir(args.output_dir)
  else:
    logger.info('update saved index')
    filepath = os.path.join(args.output_dir, 'image%d.jpg' % (saved_index))
    while os.path.isfile(filepath):
      saved_index += 1
      filepath = os.path.join(args.output_dir, 'image%d.jpg' % (saved_index))
  if os.path.isfile(args.checkpoint + '.meta') and \
      os.path.isfile(args.checkpoint + '.index'):
    with tf.Session() as sess:
      model.load(sess, args.checkpoint)

      logger.info('opening camera')
      cams = []
      for i in range(args.num_camera):
        try:
          cams.append(cv2.VideoCapture(i))
        except:
          continue
      while True:
        for c, cam in enumerate(cams):
          images = []
          if cam.isOpened():
            _, image = cam.read()
            images.append(image)
            cv2.imshow('Camera image %d' % c, image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            input_image = np.reshape(cv2.resize(gray,
              (args.input_width, args.input_height)),
              [1, args.input_height, args.input_width, 1])
            grid = np.squeeze(model.predict(sess, input_image), axis=0)
            indicator = grid[:, :, NUM_OUTPUT_CLASSES] > args.threshold

            indicator_display = indicator * 255
            indicator_display[indicator_display > 255] = 255
            indicator_display = indicator_display.astype(np.uint8)
            #  cv2.imshow('Indicator %d' % i, indicator_display)
            #  print('\rmax score: %f' % (grid[:, :, NUM_OUTPUT_CLASSES].max()),
            #    end='')

            roi_display = np.copy(image)
            rois = grid[:, :, NUM_OUTPUT_CLASSES:]
            for i in range(NUM_OUTPUT_CLASSES):
              roi = rois[indicator, :]
              classes = grid[indicator, :NUM_OUTPUT_CLASSES]
              for j in range(len(roi)):
                cx = roi[j, 1] * 640
                cy = roi[j, 2] * 480
                width = roi[j, 3] * 640
                height = roi[j, 4] * 480
                p1 = (int(cx - width / 2), int(cy - height / 2))
                p2 = (int(cx + width / 2), int(cy + height / 2))
                class_index = np.argmax(classes[j, :])
                cv2.rectangle(roi_display, p1, p2, (0, 255, 0), 2)
                cv2.putText(roi_display, OUTPUT_CHARACTERS[class_index], p1,
                  cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))
            cv2.imshow('ROI %d' % c, roi_display)

        key = cv2.waitKey(10)
        if key in [ord('q'), ord('Q'), 10, 27]:
          break
        elif key in [ord('S'), ord('s')]:
          for img in images:
            logger.info('save image %d...' % (saved_index))
            output_image_path = os.path.join(args.output_dir,
              'image%d.jpg' % (saved_index))
            saved_index += 1
            cv2.imwrite(output_image_path, img)


if __name__ == '__main__':
  main()

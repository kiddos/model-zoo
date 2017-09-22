import tensorflow as tf
import numpy as np
import cv2
import logging
import os
from argparse import ArgumentParser

from yolo_face_v3 import YOLOFace, IMAGE_WIDTH, IMAGE_HEIGHT


logging.basicConfig()
logger = logging.getLogger('yolo face v3 inference')
logger.setLevel(logging.INFO)


def main():
  parser = ArgumentParser()
  parser.add_argument('--checkpoint', dest='checkpoint',
    default='yolo_face_v3_0/yolo_face-200000', help='checkpoint to load')
  parser.add_argument('--threshold', dest='threshold',
    default=0.3, type=float, help='threshold for indicator')
  args = parser.parse_args()

  if os.path.isfile(args.checkpoint + '.index') and \
      os.path.isfile(args.checkpoint + '.meta'):

    model = YOLOFace()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      saver.restore(sess, args.checkpoint)

      camera = cv2.VideoCapture(0)
      if camera.isOpened():
        while True:
          _, image = camera.read()
          input_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
          indicator, coord, size = sess.run(
              [model.indicator_output, model.coord_output,
                model.size_output], feed_dict={
                  model.images: np.expand_dims(input_image, axis=0),
                  model.keep_prob: 1.0})
          valid = np.squeeze(indicator > args.threshold)
          coordinates = coord[0, valid, :]
          sizes = size[0, valid, :]
          assert len(coordinates) == len(sizes)
          print(valid.shape, coordinates.shape, sizes.shape)
          for i in range(len(coordinates)):
            x = coordinates[i, 0] * image.shape[1]
            y = coordinates[i, 1] * image.shape[0]
            w = sizes[i, 0] * image.shape[1] * image.shape[1] / IMAGE_WIDTH
            h = sizes[i, 1] * image.shape[0] * image.shape[0] / IMAGE_HEIGHT
            p1 = (int(x - w / 2), int(y - h / 2))
            p2 = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(image, p1, p2, (100, 255, 100), 2)

          indicator_display = cv2.resize(
            np.squeeze(valid * 255.0).astype(np.uint8), (300, 300))

          cv2.imshow('input image', image)
          cv2.imshow('indicator', indicator_display)
          key = cv2.waitKey(10)
          if key in [ord('q'), ord('Q'), 27]:
            break
  else:
    logger.warn('No such checkpoint: %s' % (args.checkpoint))



if __name__ == '__main__':
  main()

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import logging
import sys
import cv2
from PIL import Image
from argparse import ArgumentParser

logging.basicConfig()
logger = logging.getLogger('yolo')
logger.setLevel(logging.INFO)


def pad_to_fit(image, size):
  w, h = image.shape[1], image.shape[0]
  output_image = np.zeros(shape=[size, size, 3], dtype=np.uint8)
  if w >= h:
    scale = float(size) / w
    scale_h = int(h * scale)
    offset = (size - scale_h) / 2
    output_image[offset:offset+scale_h, :] = cv2.resize(image, (size, scale_h))
  else:
    scale = float(size) / h
    scale_w = int(w * scale)
    offset = (size - scale_w) / 2
    output_image[:, offset:offset+scale_w] = cv2.resize(image, (scale_w, size))
  return output_image


def convert_to_inputs(image):
  return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)


def intersect(c1, c2):
  left = np.max([c1[0] - c1[2] / 2, c2[0] - c2[2] / 2])
  right = np.min([c1[0] + c1[2] / 2, c2[0] + c2[2] / 2])
  top = np.min([c1[1] - c1[3] / 2, c2[1] + c2[3] / 2])
  bot = np.max([c1[1] - c1[3] / 2, c2[1] + c2[3] / 2])
  return left <= right and top <= bot


def draw_bounding_box(image, result, threshold, padding):
  def flatten(result, index):
    r = result[:, :, index::5]
    return np.reshape(r, [r.shape[0] * r.shape[1] * r.shape[2]])

  p = flatten(result, 0)
  axis = np.argsort(p)[::-1]
  p = p[axis]

  valid = p >= threshold
  print('\rbounding box: %d' % (np.sum(valid.astype(np.float32))), end='')
  sys.stdout.flush()

  for b in range(len(p)):
    if p[b] > threshold:
      x = flatten(result, 1)[axis[b]]
      y = flatten(result, 2)[axis[b]]
      w = flatten(result, 3)[axis[b]]
      h = flatten(result, 4)[axis[b]]

      p1 = (int(x - w / 2 - padding), int(y - w / 2 - padding))
      p2 = (int(x + w / 2 + padding), int(y + w / 2 + padding))
      cv2.rectangle(image, p1, p2, (0, 255, 0), thickness=3)

      for b2 in range(b, len(p)):
        if p[b2] > threshold:
          x2 = flatten(result, 1)[axis[b2]]
          y2 = flatten(result, 2)[axis[b2]]
          w2 = flatten(result, 3)[axis[b2]]
          h2 = flatten(result, 4)[axis[b2]]

          if intersect([x, y, w, h], [x2, y2, w2, h2]):
            p[b2] = 0.0


def run(graph, args):
  images = graph.get_tensor_by_name('import/input_images:0')
  keep_prob = graph.get_tensor_by_name('import/keep_prob:0')
  try:
    output = graph.get_tensor_by_name('import/yolo/prediction:0')
  except:
    output = graph.get_tensor_by_name('import/yolo_1/prediction:0')

  image_size = images.get_shape().as_list()[1]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=graph, config=config) as sess:
    try:
      camera = cv2.VideoCapture(0)
      if camera.isOpened():
        while True:
          _, img = camera.read()
          scaled_image = pad_to_fit(img, image_size)
          input_image = convert_to_inputs(scaled_image)

          result = sess.run(output, feed_dict={
            images: input_image,
            keep_prob: 1.0,
          })[0]

          draw_bounding_box(scaled_image, result, args.threshold, args.padding)

          display_image = cv2.resize(scaled_image,
            (args.display_size, args.display_size))

          cv2.imshow('Camera', display_image)

          key = cv2.waitKey(20)
          if key == ord('q'):
            break
    except Exception as e:
      logger.error(e.message)
      logger.warning('exit')


def load_frozen_graph(graph):
  with tf.gfile.GFile(graph, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as g:
    tf.import_graph_def(graph_def)
  return g


def main():
  parser = ArgumentParser()
  parser.add_argument('--model', dest='model', type=str,
    default='yolo.pb', help='model.pb path')
  parser.add_argument('--threshold', dest='threshold', type=float,
    default=0.8, help='threshold for indicator that there is an object')
  parser.add_argument('--display-size', dest='display_size', type=int,
    default=512, help='display size')
  parser.add_argument('--padding', dest='padding', type=int,
    default=20, help='padding for bounding box')

  args = parser.parse_args()

  if os.path.isfile(args.model):
    graph = load_frozen_graph(args.model)
    run(graph, args)
  else:
    logger.error('%s not found', args.model)


if __name__ == '__main__':
  main()

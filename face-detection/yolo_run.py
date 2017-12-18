import tensorflow as tf
import numpy as np
import os
import logging
from PIL import Image
from argparse import ArgumentParser

logging.basicConfig()
logger = logging.getLogger('yolo')
logger.setLevel(logging.INFO)

def fit(original, size):
  w, h = original.size
  output_image = Image.new('RGB', (size, size))
  if w >= h:
    scale = float(size) / w
    oh = int(h * scale)
    output_image.paste(original.resize((size, oh)),
      (0, (size - oh) / 2))
  else:
    scale = float(size) / h
    ow = int(w * scale)
    output_image.paste(original.resize((ow, size)),
      ((size - ow) / 2, 0))
  return np.expand_dims(np.array(output_image, np.uint8), axis=0)


def intersect(c1, c2):
  left = np.max([c1[0] - c1[2] / 2, c2[0] - c2[2] / 2])
  right = np.max([c1[0] + c1[2] / 2, c2[0] + c2[2] / 2])
  top = np.min([c1[1] - c1[3] / 2, c2[1] + c2[3] / 2])
  bot = np.min([c1[1] - c1[3] / 2, c2[1] + c2[3] / 2])
  return left <= right and top <= bot


def run(graph, output_node_name, threshold):
  images = graph.get_tensor_by_name('import/input_images:0')
  keep_prob = graph.get_tensor_by_name('import/keep_prob:0')
  try:
    output = graph.get_tensor_by_name('import/yolo/prediction:0')
  except:
    output = graph.get_tensor_by_name('import/yolo_1/prediction:0')

  image_size = images.get_shape().as_list()[1]
  with tf.Session(graph=graph) as sess:
    try:
      import cv2

      camera = cv2.VideoCapture(0)
      if camera.isOpened():
        while True:
          _, img = camera.read()
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = fit(Image.fromarray(img), image_size)

          result = sess.run(output, feed_dict={
            images: img,
            keep_prob: 1.0,
          })[0]

          p = result[:, :, 0]
          p = np.reshape(p, [p.shape[0] * p.shape[1]])
          axis = np.argsort(p)[::-1]
          p = p[axis]

          coord = result[:, :, 1:]
          coord = np.reshape(coord, [coord.shape[0] * coord.shape[1], -1])
          coord = coord[axis, :]

          img = np.squeeze(img)
          for i in range(len(p)):
            prob = p[i]

            if prob > threshold:
              c = coord[i, :]
              padding = 10
              cv2.rectangle(img,
                (int(c[0] - c[2] / 2 - padding), int(c[1] - c[3] / 2 - padding)),
                (int(c[0] + c[2] / 2 + padding), int(c[1] + c[3] / 2 + padding)),
                (0, 255, 0), thickness=3)

              # apply non-maximum suppression
              for j in range(i, len(p)):
                if p[j] > threshold:
                  c2 = coord[j, :]
                  if intersect(c, c2):
                    p[j] = 0.0

          cv2.imshow('Image', cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            (512, 512)))

          key = cv2.waitKey(10)
          if key == ord('q'):
            break
    except Exception as e:
      print(e)
      logger.info('exit')


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

  args = parser.parse_args()

  if os.path.isfile(args.model):
    graph = load_frozen_graph(args.model)
    run(graph, args.model, args.threshold)
  else:
    logger.error('%s not found', args.model)


if __name__ == '__main__':
  main()

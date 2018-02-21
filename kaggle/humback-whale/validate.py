import tensorflow as tf
import numpy as np
import os
import logging
from PIL import Image

from humback_whale_data import HumbackWhaleData


logging.basicConfig()
logger = logging.getLogger('predict')
logger.setLevel(logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_csv', 'humback-whale.csv', 'output prediction')
tf.app.flags.DEFINE_string('model', 'humback-whale.pb', 'frozen model to load')
tf.app.flags.DEFINE_string('dbname', 'humback-whale.sqlite3',
  'label mapping to load')


def parse_images(images, input_width, input_height):
  output_images = []
  for image in images:
    img = Image.fromarray(image).convert('L')
    img = img.resize([input_width, input_height])
    img = np.array(img, np.uint8)
    img = np.reshape(img, [input_height, input_width, 1])
    output_images.append(img)
  return np.array(output_images)


def load_graph():
  with tf.gfile.GFile(FLAGS.model, 'rb') as gf:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(gf.read())

  with tf.Graph().as_default() as g:
    tf.import_graph_def(graph_def)
    return g


def predict():
  graph = load_graph()

  input_images = graph.get_tensor_by_name('import/input_images:0')
  keep_prob = graph.get_tensor_by_name('import/keep_prob:0')
  prediction = graph.get_tensor_by_name('import/inference/outputs/prediction:0')

  input_shape = input_images.get_shape().as_list()
  input_width, input_height = input_shape[1:3]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=graph, config=config) as sess:
    data = HumbackWhaleData(FLAGS.dbname, input_width, input_height, 0, 0)
    logger.info('loading images...')
    data.load()

    logger.info('predicting...')
    correct_count = 0
    batch_size = 32
    for i in range(0, len(data.images), batch_size):
      image_batch = parse_images(data.images[i:(i + batch_size)],
        input_width, input_height)
      p = sess.run(prediction, feed_dict={
        input_images: image_batch,
        keep_prob: 1.0,
      })

      label_batch = np.array(data.labels[i:(i + batch_size)])
      matched = np.sum(np.equal(label_batch, np.argmax(p,
        axis=1)).astype(np.float32))
      logger.info('correct count: %d', matched)
      correct_count += matched

    logger.info('training accuracy: %f', correct_count / len(data.images))


def main(_):
  predict()


if __name__ == '__main__':
  tf.app.run()


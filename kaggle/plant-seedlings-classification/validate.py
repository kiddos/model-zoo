import tensorflow as tf
import numpy as np
import os
import pickle
import logging
from PIL import Image


logging.basicConfig()
logger = logging.getLogger('validation')
logger.setLevel(logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('temp', 'temp.pickle',
  'temp data to load for validation')
tf.app.flags.DEFINE_string('model', 'plants.pb', 'model to validate')


def load():
  if os.path.isfile(FLAGS.temp):
    with open(FLAGS.temp, 'rb') as f:
      obj = pickle.load(f)
    return obj['train_data'], obj['train_label'], \
      obj['valid_data'], obj['valid_label']
  else:
    logger.info('%s not found', FLAGS.temp)


def load_graph():
  with tf.gfile.GFile(FLAGS.model, 'rb') as gf:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(gf.read())

  with tf.Graph().as_default() as g:
    tf.import_graph_def(graph_def)
    return g


class Validator(object):
  def __init__(self):
    self.graph = load_graph()

    self.input_images = self.graph.get_tensor_by_name(
      'import/inputs/images:0')
    self.keep_prob = self.graph.get_tensor_by_name(
      'import/inputs/keep_prob:0')
    self.prediction = self.graph.get_tensor_by_name(
      'import/inference/output/prediction:0')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(graph=self.graph, config=config)

    input_shape = self.input_images.get_shape().as_list()
    self.input_width, self.input_height = input_shape[1:3]

  def _resize(self, images):
    output_images = []
    for i in range(len(images)):
      img = Image.fromarray(images[i, ...])
      img = img.resize([self.input_width, self.input_height])
      img = np.array(img, np.uint8)
      output_images.append(img)
    return np.stack(output_images, axis=0)

  def validate(self, data, label):
    count = 0
    acc = 0
    for i in range(0, len(data), 32):
      i1 = i
      i2 = min(len(data), i + 32)
      batch = self._resize(data[i1:i2, ...])
      p = self.sess.run(self.prediction, feed_dict={
        self.input_images: batch,
        self.keep_prob: 1.0,
      })

      l = label[i1:i2, ...]
      eq = (np.argmax(p, axis=1) == np.argmax(l, axis=1))
      acc += np.mean(eq.astype(np.float32))
      count += 1
    return acc / count

  def run(self):
    train_data, train_label, valid_data, valid_label = load()
    logger.info('training data: %s', str(train_data.shape))
    logger.info('training label: %s', str(train_label.shape))
    logger.info('validation data: %s', str(valid_data.shape))
    logger.info('validation label: %s', str(valid_label.shape))

    train_acc = self.validate(train_data, train_label)
    valid_acc = self.validate(valid_data, valid_label)
    logger.info('training accuracy: %f', train_acc)
    logger.info('validation accuracy: %f', valid_acc)


def main(_):
  validator = Validator()
  validator.run()


if __name__ == '__main__':
  tf.app.run()

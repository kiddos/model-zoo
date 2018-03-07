import tensorflow as tf
import numpy as np

from plant_loader import PlantLoader
from validate import load_graph


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dbname', 'plants.sqlite3', 'db to load test data')
tf.app.flags.DEFINE_string('output_csv', 'plants_prediction.csv', 'outputs')


class Predictor(object):
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
    assert self.input_width == self.input_height

    self.loader = PlantLoader(FLAGS.dbname, self.input_width)
    self.loader.load_data()

  def run(self):
    with open(FLAGS.output_csv, 'w') as f:
      f.write('file,species\n')

      label_names = self.loader.get_label_name()
      label_names = [l[0] for l in label_names]
      file_names = self.loader.get_test_files()
      self.test_data = self.loader.get_test_images()
      for i in range(0, len(self.test_data)):
        batch = self.test_data[i:(i + 1), ...]
        p = self.sess.run(self.prediction, feed_dict={
          self.input_images: batch,
          self.keep_prob: 1.0,
        })

        l = np.argmax(p, axis=1)[0]
        f.write('%s,%s\n' % (file_names[i], label_names[l]))


def predict():
  predictor = Predictor()
  predictor.run()


if __name__ == '__main__':
  predict()

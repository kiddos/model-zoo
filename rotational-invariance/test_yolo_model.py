from __future__ import print_function
from optparse import OptionParser
import tensorflow as tf
import logging
import numpy as np

from data_util.yolo_data_util import load_data, CHARACTERS
from yolo_model import YOLOModel


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_error(coordinate_prediction, coordinate_answer,
  dimension_prediction, dimension_answer,
  classes_prediction, classes_answer, threshold=0.8):
  class_accuracy = np.sum(
    np.round(classes_prediction) ==
    np.round(classes_answer)) * 100.0 / \
      classes_answer.shape[0] / classes_answer.shape[1] / classes_answer.shape[2]

  coordinate_error = np.sum(np.abs(
    coordinate_prediction[classes_prediction > threshold] -
    coordinate_answer[classes_prediction > threshold])) * 100.0 / \
      coordinate_answer.shape[0] / coordinate_answer.shape[1] / \
      coordinate_answer.shape[2]

  dimension_error = np.sum(np.abs(
    dimension_prediction[classes_prediction > threshold] -
    dimension_answer[classes_prediction > threshold])) * 100.0 / \
      dimension_answer.shape[0] / dimension_answer.shape[1] / \
      dimension_answer.shape[2]
  return class_accuracy, coordinate_error, dimension_error


def main():
  parser = OptionParser()
  parser.add_option('-c', '--checkpoint', dest='checkpoint', default='',
      help='last checkpoint to load for testing')
  parser.add_option('-n', '--model_name', dest='model_name', default='YOLO',
      help='model name for output')
  parser.add_option('-i', '--input_db', dest='input_db',
      default='yolo_train.sqlite',
      help='model name for output')
  parser.add_option('-k', '--output_width', dest='output_width', default=16,
      help='output width')
  parser.add_option('-j', '--output_height', dest='output_height', default=12,
      help='output height')
  options, args = parser.parse_args()

  if options.checkpoint == '':
    logger.warning('require a checkpoint to be specified')
    return

  output_width = int(options.output_width)
  output_height = int(options.output_height)
  images, coordinates, dimensions, classes = \
    load_data(options.input_db, CHARACTERS,
      output_width, output_height, 1)
  print(images.shape)
  print(coordinates.shape)
  print(dimensions.shape)
  print(classes.shape)

  model = YOLOModel(160, 120, 3,
      output_width, output_height, 1,
      model_name=options.model_name)

  total_class_accuracy = 0
  total_coordinate_error = 0
  total_dimension_error = 0
  with tf.Session() as sess:
    if options.checkpoint:
      logger.info('loading checkpoint...')
      model.load(sess, options.checkpoint)

      batch_size = 200
      max_epoch = images.shape[0] / batch_size
      for epoch in range(max_epoch):
        start = epoch*batch_size
        to = (epoch + 1)*batch_size
        batch_images = images[start:to, :]
        coordinate_pred, dimension_pred, class_pred = \
          model.predict(sess, batch_images)
        batch_coordinates = coordinates[start:to, :]
        batch_dimensions = dimensions[start:to, :]
        batch_classes = classes[start:to, :]

        class_accuracy, coordinate_error, dimension_error = \
          compute_error(coordinate_pred, batch_coordinates,
          dimension_pred, batch_dimensions,
          class_pred, batch_classes, threshold=0.8)
        total_class_accuracy += class_accuracy
        total_coordinate_error += coordinate_error
        total_dimension_error += dimension_error
        print('class accuracy: %f, coordinate error: %f, dimension error: %f' %
          (class_accuracy, coordinate_error, dimension_error))

      total_class_accuracy /= max_epoch
      total_coordinate_error /= max_epoch
      total_dimension_error /= max_epoch
    print('AVE class error: %f, coordinate error: %f, dimension error: %f' %
      (total_class_accuracy, total_coordinate_error, total_dimension_error))



if __name__ == '__main__':
  main()

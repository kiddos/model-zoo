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
tf.app.flags.DEFINE_integer('stride', 2, 'moving window for test images sampling')
tf.app.flags.DEFINE_string('test_images', 'test', 'test image folder')
tf.app.flags.DEFINE_string('dbname', 'humback-whale.sqlite3',
  'label mapping to load')


def load_image(image_path, input_width, input_height):
  if os.path.isfile(image_path):
    image = Image.open(image_path).convert('L')
    w, h = image.size
    cropped = []
    if w > h:
      for i in range(0, w - h + 1, FLAGS.stride):
        crop = image.crop([i, 0, i + h, h])
        cropped.append(crop)
    else:
      for i in range(0, h - w + 1, FLAGS.stride):
        crop = image.crop([0, i, w, i + w])
        cropped.append(crop)

    for i in range(len(cropped)):
      cropped[i] = cropped[i].resize([input_width, input_height])
      cropped[i] = np.expand_dims(np.array(cropped[i]), axis=3)
    return np.array(cropped)
  else:
    logger.info('image %s not found', image_path)


def load_graph():
  with tf.gfile.GFile(FLAGS.model, 'rb') as gf:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(gf.read())

  with tf.Graph().as_default() as g:
    tf.import_graph_def(graph_def)
    return g


def predict():
  if not os.path.isdir(FLAGS.test_images):
    logger.error('test images not found')
    return

  graph = load_graph()

  input_images = graph.get_tensor_by_name('import/input_images:0')
  keep_prob = graph.get_tensor_by_name('import/keep_prob:0')
  prediction = graph.get_tensor_by_name('import/inference/outputs/prediction:0')

  input_shape = input_images.get_shape().as_list()
  input_width, input_height = input_shape[1:3]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=graph, config=config) as sess:
    images = sorted(os.listdir(FLAGS.test_images))
    data = HumbackWhaleData(FLAGS.dbname, input_width, input_height)
    data.load_label_mapping()

    with open(FLAGS.output_csv, 'w') as f:
      # put first column line
      f.write('Image,Id\n')

      image_count = len(images)
      for i, image_name in enumerate(images):
        image_path = os.path.join(FLAGS.test_images, image_name)
        logger.info('%s', image_path)
        p = sess.run(prediction, feed_dict={
          input_images: load_image(image_path, input_width, input_height),
          keep_prob: 1.0,
        })
        p = np.sum(p, axis=0)
        most_likely = np.argsort(p)[::-1]

        f.write('%s,%s %s %s %s %s\n' % (image_name,
          data.label_mapping[most_likely[0]],
          data.label_mapping[most_likely[1]],
          data.label_mapping[most_likely[2]],
          data.label_mapping[most_likely[3]],
          data.label_mapping[most_likely[4]]))

        if i % (image_count / 10) == 0:
          logger.info('processed %d/%d.', i, image_count)


def main(_):
  #  images = load_image('./test/00029b3a.jpg', 32, 32)
  #  print images.shape
  predict()


if __name__ == '__main__':
  tf.app.run()

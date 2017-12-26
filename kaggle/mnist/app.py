from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
from argparse import ArgumentParser
from PIL import Image
from json import loads, dumps
import os
import sys
import coloredlogs
import logging
import numpy as np
import tensorflow as tf


coloredlogs.install()
logging.basicConfig()
logger = logging.getLogger('mnist')
logger.setLevel(logging.INFO)


if os.path.isfile('mnist.pb'):
  logger.info('loading mnist.pb...')
  with tf.gfile.GFile('mnist.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as g:
    tf.import_graph_def(graph_def)

  logger.info('initializer session...')
  session = tf.InteractiveSession()

  graph = tf.get_default_graph()
  input_images = graph.get_tensor_by_name('import/inputs/input_images:0')
  keep_prob = graph.get_tensor_by_name('import/inputs/keep_prob:0')
  prediction = graph.get_tensor_by_name('import/mnist/output/prediction:0')
else:
  logger.error('fail to load mnist.pb')
  sys.exit(0)

app = Flask('mnist')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/js/<path>')
def serve_js(path):
  return send_from_directory('js', path)

@app.route('/css/<path>')
def serve_css(path):
  return send_from_directory('css', path)

@app.route('/image', methods=['POST'])
def post_image():
  print('image')
  try:
    image_data = request.form['image']
    image_data = np.array([int(e) for e in image_data.split(',')],
      dtype=np.uint8)
    image_data = image_data.reshape([1, 300, 300, 4])[:, :, :, :3]
    p = session.run(prediction, feed_dict={
      input_images: image_data,
      keep_prob: 1.0
    })[0, :]
    return dumps(p)
  except Exception as e:
    print(e)
    return 'failed'


def main():
  parser = ArgumentParser()
  parser.add_argument('--host', dest='host', default='0.0.0.0',
    type=str, help='host to run app')
  parser.add_argument('--port', dest='port', default=5000,
    type=int, help='port to run app')
  parser.add_argument('--debug', dest='debug', default=True,
    type=bool, help='debug mode')

  args = parser.parse_args()

  app.run(args.host, args.port, debug=args.debug)


if __name__ == '__main__':
  main()

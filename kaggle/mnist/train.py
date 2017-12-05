import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mnist_prepare import load
from mnist import MNISTConvolutionModel

def main():
  parser = ArgumentParser()
  parser.add_argument('--db', dest='db', default='mnist.sqlite3',
      help='input sqlite3 db')

  args = parser.parse_args()
  train_images, train_label, test_images = load(args.db)
  print('train images: %s' % (str(train_images.shape)))
  print('train labels: %s' % (str(train_label.shape)))
  print('test images: %s' % (str(test_images.shape)))

  image_size = 28
  image_channel = 1
  model = MNISTConvolutionModel(image_size, image_size, image_channel,
    10, model_name='MNIST', saving=True)

  with tf.Session() as sess:
    model.train(sess, train_images, train_label,
      batch_size=256,
      output_period=10,
      keep_prob=0.8,
      max_epoch=100000)



if __name__ == '__main__':
  main()

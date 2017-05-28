import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mnist_prepare import load

def main():
  parser = ArgumentParser()
  parser.add_argument('--db', dest='db', default='mnist.sqlite3',
      help='input sqlite3 db')

  args = parser.parse_args()
  train_images, train_label, test_images = load(args.db)
  print('train images: %s' % (str(train_images.shape)))
  print('train labels: %s' % (str(train_label.shape)))
  print('test images: %s' % (str(test_images.shape)))


if __name__ == '__main__':
  main()

import numpy as np
from sklearn.svm import SVC
from argparse import ArgumentParser

from titanic_prepare import load_data


def accuracy(prediction, label):
  return np.sum((prediction == label).astype(np.float32)) / len(prediction)

def train(args):
  training_data, training_label, valid_data, valid_label = \
    load_data(args.csv_file)

  svm = SVC()
  svm.fit(training_data, np.argmax(training_label, axis=1))
  prediction = svm.predict(training_data)
  print(accuracy(prediction, np.argmax(training_label, axis=1)))
  print(accuracy(svm.predict(valid_data), np.argmax(valid_label, axis=1)))


def main():
  parser = ArgumentParser()
  parser.add_argument('--csv-file', dest='csv_file', default='train.csv',
    type=str, help='training csv file')
  args = parser.parse_args()

  train(args)


if __name__ == '__main__':
  main()

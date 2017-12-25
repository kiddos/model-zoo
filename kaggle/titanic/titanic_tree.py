import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser

from titanic_prepare import load_data, load_test_data


def accuracy(prediction, label):
  return np.sum((prediction == label).astype(np.float32)) / len(prediction)

def train(args):
  training_data, training_label, valid_data, valid_label = \
    load_data(args.csv_file)
  test_data = load_test_data('test.csv')

  training_data = np.concatenate([training_data, valid_data], axis=0)
  training_label = np.concatenate([training_label, valid_label], axis=0)

  classifier = RandomForestClassifier()
  classifier.fit(training_data, np.argmax(training_label, axis=1))
  prediction = classifier.predict(training_data)
  print(accuracy(prediction, np.argmax(training_label, axis=1)))
  print(accuracy(classifier.predict(valid_data), np.argmax(valid_label, axis=1)))

  prediction = classifier.predict(test_data)
  with open('tree_output.csv', 'w') as f:
    f.write('PassengerId,Survived\n')

    for i in range(len(prediction)):
      f.write('%d,%d\n' % (i + 892, prediction[i]))


def main():
  parser = ArgumentParser()
  parser.add_argument('--csv-file', dest='csv_file', default='train.csv',
    type=str, help='training csv file')
  args = parser.parse_args()

  train(args)


if __name__ == '__main__':
  main()

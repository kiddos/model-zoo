from __future__ import print_function
from __future__ import absolute_import

from pandas import read_csv
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os


CLASS = 'Survived'
FEATURE_PCLASS = 'Pclass'
FEATURE_GENDER = 'Sex'
FEATURE_AGE = 'Age'
FEATURE_SIBSP = 'SibSp'
FEATURE_PARCH = 'Parch'
FEATURE_FARE = 'Fare'
FEATURE_EMBARKED = 'Embarked'


def analyse_data(titanic_data, plot=True):
  plt.rc("font", size=14)

  sb.set(style="white")
  sb.set(style="whitegrid", color_codes=True)

  print('NA data')
  print(titanic_data.isnull().sum())

  if plot:
    plt.figure()
    ax = titanic_data[FEATURE_AGE].hist(bins=15, color='purple', alpha=0.8)
    ax.set(xlabel=FEATURE_AGE, ylabel='Count')
    plt.savefig('age.png')
  titanic_data[FEATURE_AGE].fillna(
    titanic_data[FEATURE_AGE].median(skipna=True), inplace=True)

  if plot:
    plt.figure()
    sb.countplot(x=FEATURE_EMBARKED, data=titanic_data, palette='Set2')
    plt.savefig('embarked.png')
  titanic_data[FEATURE_EMBARKED].fillna('S', inplace=True)

  if plot:
    plt.figure()
    sb.countplot(x=FEATURE_GENDER, data=titanic_data, palette='Set1')
    plt.savefig('gender.png')

  if plot:
    plt.figure()
    ax = titanic_data['Fare'].hist(bins=16, color='black', alpha=0.6)
    ax.set(xlabel='Fare', ylabel='count')
    plt.savefig('fare.png')
  titanic_data['Fare'].fillna(
    titanic_data['Fare'].median(skipna=True), inplace=True)

  print('\nNA data (After processing):')
  print(titanic_data.isnull().sum())


def parse_label(titanic_data):
  output_classes = []
  for entry in titanic_data[CLASS]:
    if entry == 1:
      output_classes.append([0, 1])
    elif entry == 0:
      output_classes.append([1, 0])
  assert len(output_classes) == len(titanic_data[CLASS])
  return np.array(output_classes)


def parse_data(titanic_data):

  pclass = []
  for entry in titanic_data[FEATURE_PCLASS]:
    pclass.append(entry)
  assert len(pclass) == len(titanic_data[FEATURE_PCLASS])

  gender = []
  for entry in titanic_data[FEATURE_GENDER]:
    if entry.lower() == 'male':
      gender.append(1)
    elif entry.lower() == 'female':
      gender.append(0)
  assert len(gender) == len(titanic_data[FEATURE_GENDER])

  age = []
  for entry in titanic_data[FEATURE_AGE]:
    age.append(entry)
  assert len(age) == len(titanic_data[FEATURE_AGE])

  sibsp = []
  for entry in titanic_data[FEATURE_SIBSP]:
    sibsp.append(entry)
  assert len(sibsp) == len(titanic_data[FEATURE_SIBSP])

  parch = []
  for entry in titanic_data[FEATURE_PARCH]:
    parch.append(entry)
  assert len(parch) == len(titanic_data[FEATURE_PARCH])

  fare = []
  for entry in titanic_data[FEATURE_FARE]:
    fare.append(entry)
  assert len(fare) == len(titanic_data[FEATURE_FARE])

  embarked = []
  for entry in titanic_data[FEATURE_EMBARKED]:
    if entry == 'S':
      embarked.append([1, 0, 0])
    elif entry == 'C':
      embarked.append([0, 1, 0])
    elif entry == 'Q':
      embarked.append([0, 0, 1])

  data = []
  for entries in zip(pclass, gender, age, sibsp, parch, fare, embarked):
    data.append([e for e in entries[:-1]] + entries[-1])
  return np.array(data)


def main():
  parser = ArgumentParser()
  parser.add_argument('--csv-file', dest='csv_file',
    default='titanic_clean.csv', help='csv file to load')
  args = parser.parse_args()

  if os.path.isfile(args.csv_file):
    titanic_data = read_csv(args.csv_file)
    analyse_data(titanic_data)

    data = parse_data(titanic_data)
    for entry in data:
      print(entry)

    label = parse_label(titanic_data)
    print(label.shape)


if __name__ == '__main__':
  main()

import sqlite3
import re
import os
import numpy as np


def remove_name(line):
  pattern = re.compile(r'\"(.+?)\"')
  found = pattern.findall(line)
  if found:
    name = found[0]
  else:
    name = ''
  index = line.find(name)
  line = line[0: index] + line[index + len(name):]
  return line, name


def process_entry(entry_txt, name, starting_index):
  entry = []

  # name
  if name.find('Mr') == 0:
    entry.append(1)
    entry += [0, 1, 0, 0, 0]
  elif name.find('Mrs') == 0:
    entry.append(2)
    entry += [0, 0, 1, 0, 0]
  elif name.find('Miss') == 0:
    entry.append(3)
    entry += [0, 0, 0, 1, 0]
  elif name.find('Master') == 0:
    entry.append(4)
    entry += [0, 0, 0, 0, 1]
  else:
    entry.append(0)
    entry += [1, 0, 0, 0, 0]
  # pclass
  pclass = [0 for i in range(3)]
  pclass[int(entry_txt[starting_index]) - 1] = 1
  entry += pclass
  entry.append(int(entry_txt[starting_index]))

  # gender
  if entry_txt[starting_index + 2] == 'male':
    entry.append(1)
  elif entry_txt[starting_index + 2] == 'female':
    entry.append(0)

  # age
  if entry_txt[starting_index + 3]:
    age = float(entry_txt[starting_index + 3])
  else:
    age = 0
  entry.append(age)
  if age < 14:
    entry += [1, 0, 0]
  elif age < 32:
    entry += [0, 1, 0]
  else:
    entry += [0, 0, 1]

  # siblings
  entry.append(int(entry_txt[starting_index + 4]))
  # parents
  entry.append(int(entry_txt[starting_index + 5]))
  # is alone
  family_count = int(entry_txt[starting_index + 4]) + \
    int(entry_txt[starting_index + 5])
  if family_count == 0:
    entry.append(1)
  else:
    entry.append(0)
  #  # fare
  if entry_txt[starting_index + 7] and entry_txt[starting_index + 7] != 'NA':
    entry.append(float(entry_txt[starting_index + 7]))
  else:
    entry.append(0)

  # embarked
  if entry_txt[starting_index + 9] == 'C':
    entry += [1, 0, 0]
    entry.append(1)
  elif entry_txt[starting_index + 9] == 'Q':
    entry += [0, 1, 0]
    entry.append(2)
  elif entry_txt[starting_index + 9] == 'S':
    entry += [0, 0, 1]
    entry.append(3)
  else:
    entry += [0, 0, 0]
    entry.append(0)
  return entry

def load_train_data(csv_file):
  data = []
  label = []
  if os.path.isfile(csv_file):
    with open(csv_file, 'r') as f:
      line = f.readline()
      while True:
        line = f.readline()
        if not line: break

        line, name = remove_name(line.strip())

        entry_txt = line.split(',')
        entry = process_entry(entry_txt, name, 2)

        data.append(np.array(entry))

        if entry_txt[1] == '1':
          label.append([0, 1])
        elif entry_txt[1] == '0':
          label.append([1, 0])
  return np.array(data), np.array(label)


def load_test_data(csv_file):
  data = []
  if os.path.isfile(csv_file):
    with open(csv_file, 'r') as f:
      line = f.readline()
      while True:
        line = f.readline()
        if not line: break

        line, name = remove_name(line.strip())

        entry_txt = line.split(',')
        entry = process_entry(entry_txt, name, 1)
        data.append(entry)
  return np.array(data)


def load_data(csv_file, training_percent=0.8):
  data, label = load_train_data(csv_file)
  index = np.random.permutation(np.arange(len(data)))
  training_index = index[:int(training_percent * len(data))]
  validation_index = index[int(training_percent * len(data)):]

  training_data = data[training_index, :]
  training_label = label[training_index, :]

  validation_data = data[validation_index, :]
  validation_label = label[validation_index, :]

  return training_data, training_label, validation_data, validation_label


def main():
  data, label = load_train_data('train.csv')
  test_data = load_test_data('test.csv')

  print(data.shape)
  print(label.shape)
  print(test_data.shape)


if __name__ == '__main__':
  main()

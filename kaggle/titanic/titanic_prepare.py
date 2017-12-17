import sqlite3
import re
import os
import numpy as np


def remove_name(line):
  pattern = re.compile(r'\"(.+?)\"')
  found = pattern.findall(line)
  name = found[0]
  index = line.find(name)
  line = line[0: index] + line[index + len(name):]
  return line, name


def load_train_data(csv_file):
  data = []
  label = []
  if os.path.isfile(csv_file):
    with open(csv_file, 'r') as f:
      line = f.readline()
      while True:
        line = f.readline()
        if not line: break

        line, _ = remove_name(line.strip())

        entry_txt = line.split(',')
        entry = []
        entry.append(int(entry_txt[2]))
        if entry_txt[4] == 'male':
          entry.append(1)
        elif entry_txt[4] == 'female':
          entry.append(0)
        if entry_txt[5]:
          entry.append(float(entry_txt[5]))
        else:
          entry.append(-1)
        entry.append(int(entry_txt[6]))
        entry.append(int(entry_txt[7]))
        entry.append(float(entry_txt[9]))

        if entry_txt[11] == 'C':
          entry += [1, 0, 0]
        elif entry_txt[11] == 'Q':
          entry += [0, 1, 0]
        elif entry_txt[11] == 'S':
          entry += [0, 0, 1]
        else:
          entry += [0, 0, 0]

        assert len(entry) == 9
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

        line, _ = remove_name(line.strip())

        entry_txt = line.split(',')
        entry = []
        entry.append(int(entry_txt[1]))
        if entry_txt[3] == 'male':
          entry.append(1)
        elif entry_txt[3] == 'female':
          entry.append(0)

        if entry_txt[4]:
          entry.append(float(entry_txt[4]))
        else:
          entry.append(-1)
        entry.append(int(entry_txt[5]))
        entry.append(int(entry_txt[6]))

        if entry_txt[8]:
          entry.append(float(entry_txt[8]))
        else:
          entry.append(0.0)

        if entry_txt[10] == 'C':
          entry += [1, 0, 0]
        elif entry_txt[10] == 'Q':
          entry += [0, 1, 0]
        elif entry_txt[10] == 'S':
          entry += [0, 0, 1]

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

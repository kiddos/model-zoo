import numpy as np
import sqlite3
import logging
from PIL import Image, ImageDraw
from argparse import ArgumentParser

logging.basicConfig()
logger = logging.getLogger('wider')
logger.setLevel(logging.INFO)

class WIDERLoader(object):
  def __init__(self, dbname):
    self.dbname = dbname
    self.connection = sqlite3.connect(dbname)
    self.cursor = self.connection.cursor()

    logger.info('loading meta data...')
    self._load_meta()

  def __del__(self):
    logger.info('closing connection...')
    self.connection.close()

  def _load_meta(self):
    self.cursor.execute("""SELECT * FROM meta;""")
    meta = self.cursor.fetchall()[0]
    self.input_size = meta[0]
    self.output_size = meta[1]

  def load_data(self):
    self.load_training_data()

  def _load_data_from_table(self, table):
    logger.info('loading data from %s...', table)
    data = []
    label = []

    self.cursor.execute("""SELECT image, label FROM %s;""" % (table))
    raw_data = self.cursor.fetchall()
    for entry in raw_data:
      input_image = np.frombuffer(entry[0], np.uint8).reshape([
        self.input_size, self.input_size, 3])
      label = np.frombuffer(entry[1], np.float32).reshape([
        self.output_size, self.output_size, 5])
      data.append(input_image)
      label.append(label)
    return np.array(data), np.array(label)

  def load_training_data(self):
    self.training_data, self.training_label = \
      self._load_data_from_table('wider_train')

  def load_validation_data(self):
    self.validation_data, self.validation_label = \
      self._load_data_from_table('wider_valid')

  def get_input_size(self):
    return self.input_size

  def get_output_size(self):
    return self.output_size

  def get_training_data(self):
    index = np.random.permutation(np.arange(len(self.training_data)))
    return self.training_data[index, :], self.training_label[index, :]

  def get_validation_data(self):
    index = np.random.permutation(np.arange(len(self.validation_data)))
    return self.validation_data[index, :], self.validation_label[index, :]


def main():
  parser = ArgumentParser()
  parser.add_argument('--dbname', dest='dbname', type=str,
    default='wider.sqlite3', help='sqlite3 db to load')
  args = parser.parse_args()

  loader = WIDERLoader(args.dbname)
  loader.load_data()

  logger.info('training data shape: %s' % (str(loader.training_data.shape)))
  logger.info('training label shape: %s' % (str(loader.training_label.shape)))
  assert len(loader.training_data) == len(loader.training_label)

  test_index = np.random.choice(np.arange(len(loader.training_data)))
  test_image = loader.training_data[test_index, :]
  test_label = loader.training_label[test_index, :]

  output = Image.fromarray(test_image)
  draw = ImageDraw.Draw(output)
  for r in range(loader.output_size):
    for c in range(loader.output_size):
      p = test_label[r, c, 0]
      if p == 1.0:
        logger.info('face at (%d, %d) grid cell' % (c, r))
        cx = test_label[r, c, 1]
        cy = test_label[r, c, 2]
        w = test_label[r, c, 3]
        h = test_label[r, c, 4]
        x = int((cx - w / 2) * loader.input_size)
        y = int((cy - h / 2) * loader.input_size)
        w = int(w * loader.input_size)
        h = int(h * loader.input_size)

        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0))
  del draw
  output.save('test_image.jpg')


if __name__ == '__main__':
  main()

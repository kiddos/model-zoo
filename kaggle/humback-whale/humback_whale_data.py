from __future__ import print_function

from PIL import Image
import numpy as np
import sqlite3
import os
import random


class HumbackWhaleData(object):
  def __init__(self, dbname, image_width, image_height, random_padding):
    self.image_width, self.image_height = image_width, image_height
    self.random_padding = random_padding

    if os.path.isfile(dbname):
      self.connection = sqlite3.connect(dbname)
      self.cursor = self.connection.cursor()
    else:
      print('%s not found' % (dbname))

  def load(self):
    self.cursor.execute("""SELECT i.image, i.width, i.height, l.label
      FROM images i, labels l WHERE i.label == l.name;""")
    raw_data = self.cursor.fetchall()

    self.images = []
    self.labels = []
    for entry in raw_data:
      img = np.frombuffer(entry[0], dtype=np.uint8)
      img = np.reshape(img, [entry[2], entry[1]])
      self.images.append(img)
      self.labels.append(entry[3] - 1)

    self.labels = np.array(self.labels)

  def load_label_mapping(self):
    self.cursor.execute("""SELECT * FROM labels;""")
    self.label_mapping = {label - 1: name
      for label, name in self.cursor.fetchall()}

  def random_preprocess(self, image_data):
    image = Image.fromarray(image_data)
    w, h = image.size
    if self.random_padding * 2 < w:
      left = random.randint(0, self.random_padding)
      right = random.randint(w - self.random_padding, w)
    else:
      left = 0
      right = w

    if self.random_padding * 2 < h:
      top = random.randint(0, self.random_padding)
      bot = random.randint(h - self.random_padding, h)
    else:
      top = 0
      bot = h

    cropped = image.crop([left, top, right, bot])

    resized = cropped.resize([self.image_width, self.image_height],
      Image.NEAREST)
    resized = np.array(resized, np.uint8)
    return np.expand_dims(resized, axis=3)

  def get_batch(self, batch_size):
    index = np.random.permutation(np.arange(len(self.images)))[:batch_size]
    batch_images = []
    batch_label = []
    for i in index:
      img = self.random_preprocess(self.images[i])
      batch_images.append(img)
      batch_label.append(self.labels[i])
    return np.array(batch_images, dtype=np.uint8), np.array(batch_label)


def main():
  data = HumbackWhaleData('./humback-whale.sqlite3', 64, 64, 20)
  data.load()
  data.load_label_mapping()

  images, labels = data.get_batch(256)
  print(data.labels.max())
  print(images.shape)
  print(labels.shape)

  img = Image.fromarray(images[0, :, :, 0])
  img.save('test.png')


if __name__ == '__main__':
  main()

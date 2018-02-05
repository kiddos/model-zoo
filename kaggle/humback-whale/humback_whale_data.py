from __future__ import print_function

from PIL import Image
import numpy as np
import sqlite3
import os
import random


class HumbackWhaleData(object):
  def __init__(self, dbname, image_width, image_height):
    self.image_width, self.image_height = image_width, image_height

    if os.path.isfile(dbname):
      self.connection = sqlite3.connect(dbname)
      self.cursor = self.connection.cursor()

      self._load()
    else:
      print('%s not found' % (dbname))

  def _load(self):
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

  def random_preprocess(self, image_data):
    image = Image.fromarray(image_data)
    w, h = image.size
    if w > h:
      random_x = random.randint(0, w - h)
      cropped = image.crop([random_x, 0, random_x + h, h])
    else:
      random_y = random.randint(0, h - w)
      cropped = image.crop([0, random_y, w, random_y + w])

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
  data = HumbackWhaleData('./humback-whale.sqlite3', 32, 32)
  images, labels = data.get_batch(256)
  print(data.labels.max())
  print(images.shape)
  print(labels.shape)


if __name__ == '__main__':
  main()

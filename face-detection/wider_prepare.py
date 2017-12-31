import numpy as np
import os
import logging
import sqlite3
from PIL import Image, ImageDraw
from argparse import ArgumentParser

logging.basicConfig()
logger = logging.getLogger('wider')
logger.setLevel(logging.INFO)


def load_labels(label_file):
  labels = {}
  with open(label_file, 'r') as f:
    while True:
      filename = f.readline().strip()
      if not filename: break
      labels[filename] = []
      count = int(f.readline().strip())
      for i in range(count):
        box = f.readline().strip().split(' ')
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        blur = int(box[4])
        #  expression = int(box[5])
        #  illumination = int(box[6])
        invalid = int(box[7])
        #  occlusion = int(box[8])
        #  pose = int(box[9])

        if blur < 2 and not invalid:
          labels[filename].append({'x': x, 'y': y, 'w': w, 'h': h})

    count = 0
    for l in labels:
      count += len(labels[l])

    logger.info('loaded %d images labels.' % (len(labels)))
    logger.info('total of %d bounding boxes.' % (count))
    return labels


def fit_image(original, output_width, output_height, labels):
  w, h = original.size

  output_image = Image.new('RGB', (output_width, output_height))
  #  draw = ImageDraw.Draw(output_image)
  if w >= h:
    scale = float(output_width) / w
    oh = int(h * scale)
    output_image.paste(original.resize((output_width, oh)),
      (0, (output_height - oh) / 2))
    for i in range(len(labels)):
      labels[i]['x'] = int(labels[i]['x'] * scale)
      labels[i]['y'] = int(labels[i]['y'] * scale + (output_height - oh) / 2)
      labels[i]['w'] = int(labels[i]['w'] * scale)
      labels[i]['h'] = int(labels[i]['h'] * scale)

      #  draw.rectangle([labels[i]['x'], labels[i]['y'],
      #    labels[i]['x'] + labels[i]['w'],
      #    labels[i]['y'] + labels[i]['h']],
      #    outline=(0, 255, 0))
  else:
    scale = float(output_height) / h
    ow = int(w * scale)
    output_image.paste(original.resize((ow, output_height)),
      ((output_width - ow) / 2, 0))
    for i in range(len(labels)):
      labels[i]['x'] = int(labels[i]['x'] * scale + (output_width - ow) / 2)
      labels[i]['y'] = int(labels[i]['y'] * scale)
      labels[i]['w'] = int(labels[i]['w'] * scale)
      labels[i]['h'] = int(labels[i]['h'] * scale)

      #  draw.rectangle([labels[i]['x'], labels[i]['y'],
      #    labels[i]['x'] + labels[i]['w'],
      #    labels[i]['y'] + labels[i]['h']],
      #    outline=(0, 255, 0))

  #  del draw
  #  output_image.save('image.jpg')
  return output_image, labels

def fit(original, output_width, output_height):
  w, h = original.size
  output_image = Image.new('RGB', (output_width, output_height))
  if w >= h:
    scale = float(output_width) / w
    oh = int(h * scale)
    output_image.paste(original.resize((output_width, oh)),
      (0, (output_height - oh) / 2))
  else:
    scale = float(output_height) / h
    ow = int(w * scale)
    output_image.paste(original.resize((ow, output_height)),
      ((output_width - ow) / 2, 0))
  return output_image


class WIDERData(object):
  def __init__(self, args):
    self.input_size = args.input_size
    self.output_size = args.output_size
    self.num_bounding_box = args.num_bounding_box

    self.training_image_folder = args.training_folder
    self.training_label = args.training_label
    self.valid_image_folder = args.valid_folder
    self.valid_label = args.valid_label

    self.connection = sqlite3.connect(args.dbname)
    self.cursor = self.connection.cursor()

    self._setup_tables()

  def _setup_tables(self):
    logger.info('creating meta table...')
    self.cursor.execute("""DROP TABLE IF EXISTS meta;""")
    self.cursor.execute("""CREATE TABLE meta(
      inputSize INTEGER NOT NULL,
      outputSize INTEGER NOT NULL,
      numBoundingBox INTEGER NOT NULL);""")
    self.cursor.execute("""INSERT INTO meta VALUES(?, ?, ?);""",
      [self.input_size, self.output_size, self.num_bounding_box])

    logger.info('creating training data table...')
    self.cursor.execute("""DROP TABLE IF EXISTS wider_train;""")
    self.cursor.execute("""CREATE TABLE wider_train(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL,
        label BLOB NOT NULL);""")

    logger.info('creating validation data table...')
    self.cursor.execute("""DROP TABLE IF EXISTS wider_valid;""")
    self.cursor.execute("""CREATE TABLE wider_valid(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL,
        label BLOB NOT NULL);""")

    logger.info('creating test data table...')
    self.cursor.execute("""DROP TABLE IF EXISTS wider_test;""")
    self.cursor.execute("""CREATE TABLE wider_test(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL);""")

    self.connection.commit()

  def prepare(self):
    logger.info('loading training data...')
    training_labels = load_labels(self.training_label)
    self.load_data_with_label(self.training_image_folder,
      training_labels, 'wider_train')

    logger.info('loading validation data...')
    validation_labels = load_labels(self.valid_label)
    self.load_data_with_label(self.valid_image_folder,
      validation_labels, 'wider_valid')

    logger.info('loading test data...')
    self.load_data(self.valid_image_folder, 'wider_test')

  def load_data_with_label(self, image_folder, labels, table_name):
    category = os.listdir(image_folder)
    for cat in category:
      folder_path = os.path.join(image_folder, cat)
      image_paths = os.listdir(folder_path)
      logger.info('loading %s...' % (cat))
      for p in image_paths:

        img_file = os.path.join(cat, p)
        img_path = os.path.join(folder_path, p)

        label = labels[img_file]

        img = Image.open(img_path)
        input_image, new_label = fit_image(img,
          self.input_size, self.input_size, label)

        label_image = np.zeros(dtype=np.float32,
          shape=[self.output_size, self.output_size, 5 * self.num_bounding_box])
        for l in new_label:
          x = float(l['x']) / self.input_size
          y = float(l['y']) / self.input_size
          w = float(l['w']) / self.input_size
          h = float(l['h']) / self.input_size
          cx = x + w / 2
          cy = y + h / 2

          grid_x = int(cx * self.output_size)
          grid_y = int(cy * self.output_size)

          for b in range(self.num_bounding_box):
            if label_image[grid_y, grid_x, b * 5] == 0:
              label_image[grid_y, grid_x, b * 5] = 1.0
              label_image[grid_y, grid_x, b * 5 + 1] = cx
              label_image[grid_y, grid_x, b * 5 + 2] = cy
              label_image[grid_y, grid_x, b * 5 + 3] = w
              label_image[grid_y, grid_x, b * 5 + 4] = h
              break

        self.cursor.execute("""INSERT INTO %s(
          image, label) VALUES(?, ?)""" % (table_name),
          [buffer(np.array(input_image, dtype=np.uint8)),
            buffer(label_image)])
        del img

    logger.info('commit data')
    self.connection.commit()

  def load_data(self, image_folder, table_name):
    category = os.listdir(image_folder)
    for cat in category:
      folder_path = os.path.join(image_folder, cat)
      image_paths = os.listdir(folder_path)
      logger.info('loading %s...' % (cat))
      for p in image_paths:

        img_path = os.path.join(folder_path, p)

        img = Image.open(img_path)
        input_image = fit(img,
          self.input_size, self.input_size)

        self.cursor.execute("""INSERT INTO %s(
          image) VALUES(?)""" % (table_name),
          [buffer(np.array(input_image, dtype=np.uint8))])
        del img

    logger.info('commit data')
    self.connection.commit()


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-size', dest='input_size', type=int,
    default=256, help='consistant input image size')
  parser.add_argument('--output-size', dest='output_size', type=int,
    default=19, help='consistant output image size')
  parser.add_argument('--dbname', dest='dbname', type=str,
    default='wider.sqlite3', help='sqlite3 db to save to')
  parser.add_argument('--num-bounding-box', dest='num_bounding_box', type=int,
    default=1, help='number of bounding box for each output grid cell')

  parser.add_argument('--training-folder', dest='training_folder',
    default='WIDER_train/images', type=str,
    help='folders where training images are')
  parser.add_argument('--training-label', dest='training_label',
    default='wider_face_split/wider_face_train_bbx_gt.txt', type=str,
    help='label file for training images')

  parser.add_argument('--validation-folder', dest='valid_folder',
    default='WIDER_val/images', type=str,
    help='folders where validation images are')
  parser.add_argument('--validation-label', dest='valid_label',
    default='wider_face_split/wider_face_val_bbx_gt.txt', type=str,
    help='label file for validation images')

  args = parser.parse_args()

  #  save(args)
  data = WIDERData(args)
  if os.path.isdir(args.training_folder):
    data.prepare()


if __name__ == '__main__':
  main()

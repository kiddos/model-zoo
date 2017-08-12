import sqlite3
import os
import logging
import ast
import numpy as np
import cv2
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_WIDTH = 32
OUTPUT_HEIGHT = 24
OUTPUT_CHARACTERS = [
  'H', 'I', 'W', 'N', 'R', 'O', 'B', 'T',
  '2', '0', '1', '7', 'others',
]
NUM_OUTPUT_CLASSES = len(OUTPUT_CHARACTERS)
NUM_BOUNDING_BOX = 1


def load_raw_data(db_name, table_name):
  if os.path.isfile(db_name):
    logger.info('loading from %s...' % (db_name))
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM %s;""" % (table_name))
    raw_data = cursor.fetchall()
    connection.close()
    return raw_data
  else:
    logger.info('ERROR: Fail to find %s' % (db_name))


def create_grid(image, label):
  label = ast.literal_eval(label)
  grid = np.zeros(shape=[OUTPUT_HEIGHT, OUTPUT_WIDTH, NUM_OUTPUT_CLASSES +
    4 * NUM_BOUNDING_BOX + 1])
  width_scale = OUTPUT_WIDTH * 1.0 / image.shape[1]
  height_scale = OUTPUT_HEIGHT * 1.0 / image.shape[0]
  counter = np.zeros(shape=[OUTPUT_HEIGHT, OUTPUT_WIDTH], dtype=np.int32)
  for roi in label:
    cx = (roi[0] + roi[2]) / 2.0
    cy = (roi[1] + roi[3]) / 2.0
    width = np.abs(roi[0] - roi[2])
    height = np.abs(roi[1] - roi[3])

    x = int(np.min([np.round(cx * width_scale), OUTPUT_WIDTH - 1]))
    y = int(np.min([np.round(cy * height_scale), OUTPUT_HEIGHT - 1]))
    class_index = np.argmax(np.array(OUTPUT_CHARACTERS) == roi[4])
    if class_index == NUM_OUTPUT_CLASSES - 1:
      continue
    index = counter[y, x]
    grid[y, x, class_index] = 1.0
    grid[y, x, NUM_OUTPUT_CLASSES + 5 * index] = 1.0
    grid[y, x, NUM_OUTPUT_CLASSES + 1 + 5 * index] = cx * 1.0 / image.shape[1]
    grid[y, x, NUM_OUTPUT_CLASSES + 2 + 5 * index] = cy * 1.0 / image.shape[0]
    grid[y, x, NUM_OUTPUT_CLASSES + 3 + 5 * index] = width * 1.0 / image.shape[1]
    grid[y, x, NUM_OUTPUT_CLASSES + 4 + 5 * index] = height * 1.0 / image.shape[0]
    counter[y, x] += 1
  return grid


def create_training_db(output_db, table_name, raw_data):
  logger.info('create output sqlite3 database...')
  connection = sqlite3.connect(output_db)
  cursor = connection.cursor()
  cursor.execute("""CREATE TABLE IF NOT EXISTS %s(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB NOT NULL,
    imageWidth INTEGER NOT NULL,
    imageHeight INTEGER NOT NULL,
    grid BLOB NOT NULL,
    gridWidth INTEGER NOT NULL,
    gridHeight INTEGER NOT NULL,
    gridChannel INTEGER NOT NULL);""" % (table_name))
  cursor.execute("""CREATE TABLE IF NOT EXISTS %sMeta(
    numClass INTEGER NOT NULL,
    numBoundingBox INTEGER NOT NULL);""" % (table_name))
  cursor.execute("""SELECT count(*) FROM %sMeta""" % (table_name))
  if cursor.fetchall()[0] == 0:
    logger.info('create meta data...')
    cursor.execute("""INSERT INTO %sMeta(numClasses, numBoundingBox)
      VALUES(?, ?)""", (NUM_OUTPUT_CLASSES, NUM_BOUNDING_BOX))
  logger.info('saving entries...')
  for i in range(len(raw_data)):
    entry = raw_data[i]
    image = np.frombuffer(entry[1], np.uint8).reshape([entry[3], entry[2], 3])
    grid = create_grid(image, entry[4], NUM_BOUNDING_BOX)
    cursor.execute(("""INSERT INTO %s(image, imageWidth, imageHeight,
      grid, gridWidth, gridHeight, gridChannel) VALUES(?, ?, ?, ?, ?, ?, ?)""" %
      (table_name)), (buffer(image.reshape(-1)), image.shape[1], image.shape[0],
        buffer(grid.reshape(-1)), grid.shape[1], grid.shape[0],
        grid.shape[2]))
  connection.commit()
  connection.close()


class TinyYOLODataBatch(object):
  def __init__(self, db_name, table_name, input_width, input_height):
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM %s;""" % (table_name))
    raw_data = cursor.fetchall()
    self.input_images = []
    self.grid_label = []
    for i in range(len(raw_data)):
      entry = raw_data[i]
      print(entry)
      image = np.frombuffer(entry[1], np.uint8).reshape([entry[3], entry[2], 3])
      gray = np.expand_dims(cv2.cvtColor(cv2.resize(image, (input_width, input_height)),
        cv2.COLOR_BGR2GRAY), -1)
      self.input_images.append(gray)
      grid = np.frombuffer(entry[4], np.float64).reshape([entry[6], entry[5],
        entry[7]])
      self.grid_label.append(grid)
    self.input_images = np.array(self.input_images)
    self.grid_label = np.array(self.grid_label)
    self.data_size = len(self.input_images)

  def batch(self, batch_size=256):
    batch_size = np.min([batch_size, self.data_size])
    indexes = np.random.permutation(np.arange(self.data_size))[:batch_size]
    return self.input_images[indexes, :], self.grid_label[indexes, :]


def main():
  parser = ArgumentParser()
  parser.add_argument('-i', dest='db_name', default='duckymomo.sqlite3',
    help='input database path')
  parser.add_argument('-o', dest='output_db', default='tiny_yolo_v2.sqlite3',
    help='output sqlite3 database path')
  parser.add_argument('--table-name', dest='table_name',
    default='blocks', help='output database name')
  parser.add_argument('--mode', dest='mode',
    default='test', help='mode (test/create)')
  args = parser.parse_args()

  if args.mode == 'create':
    raw_data = load_raw_data(args.db_name, args.table_name)
    create_training_db(args.output_db, args.table_name, raw_data)
  else:
    batcher = TinyYOLODataBatch(args.output_db, args.table_name,
      320, 240)
    display_width, display_height = 640, 480
    while True:
      batched_image, batch_label = batcher.batch()
      image = batched_image[0, :]
      cv2.imshow('Input Image', cv2.resize(image,
        (display_width, display_height)))

      grid = batch_label[0, :]
      roi_label = grid[:, :, NUM_OUTPUT_CLASSES:]
      roi_display = np.copy(image)
      for i in range(NUM_BOUNDING_BOX):
        indicator = roi_label[:, :, i * 5] > 0
        grid_display = (indicator * 255).astype(np.uint8)
        cv2.imshow('Label Grid %d' % (i), cv2.resize(grid_display,
          (display_width, display_height)))
        rois = roi_label[indicator]
        classes = grid[indicator, :NUM_OUTPUT_CLASSES]
        for j in range(len(rois)):
          roi = rois[j, :]
          x = roi[i * 5 + 1] * roi_display.shape[1]
          y = roi[i * 5 + 2] * roi_display.shape[0]
          w = roi[i * 5 + 3] * roi_display.shape[1]
          h = roi[i * 5 + 4] * roi_display.shape[0]
          logger.info('roi %d. %dth box x: %f | y: %f | w: %f | h: %f' %
            (j, i, x, y, w, h))
          p1 = (int(x - w / 2), int(y - h / 2))
          p2 = (int(x + w / 2), int(y + h / 2))
          cv2.rectangle(roi_display, p1, p2, (255, 255, 255), 1)
          cv2.putText(roi_display,
            OUTPUT_CHARACTERS[np.argmax(classes[j, :])], p1,
            cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
      cv2.imshow('ROI Image', cv2.resize(roi_display,
        (display_width, display_height)))
      key = cv2.waitKey(0)
      if key in [ord('q'), ord('Q'), 27]:
        break


if __name__ == '__main__':
  main()

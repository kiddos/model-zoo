import sqlite3
import threading
import logging
import json
import os
import cv2
import time
import numpy as np
from PIL import Image, ImageTk
from argparse import ArgumentParser
from collections import deque

from roi_util import threshold_characters


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Saver(object):
  def __init__(self, output_db, table_name):
    self.output_db = output_db
    self.table_name = table_name
    self.entries = deque()

  def run(self):
    connection = sqlite3.connect(self.output_db)
    cursor = connection.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS %s(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      image BLOB NOT NULL,
      imageWidth INTEGER NOT NULL,
      imageHeight INTEGER NOT NULL,
      imageChannel INTEGER NOT NULL,
      roi TEXT NOT NULL);""" %
        (self.table_name))
    self.running = True
    while self.running:
      if len(self.entries) > 0:
        entry = self.entries.popleft()
        cursor.execute("""INSERT INTO %s(image,
          imageWidth, imageHeight, imageChannel, roi) VALUES(?, ?, ?, ?, ?)"""
          % (self.table_name),
          entry)
    logger.info('flusing data...')
    while len(self.entries) > 0:
      entry = self.entries.popleft()
      cursor.execute("""INSERT INTO %s(image,
        imageWidth, imageHeight, imageChannel, roi) VALUES(?, ?, ?, ?, ?)""" %
        (self.table_name),
        entry)
    logger.info('closing output db %s' % (self.output_db))
    connection.commit()
    connection.close()

  def start(self):
    self.task = threading.Thread(target=self.run)
    self.task.start()

  def save_entry(self, image, label):
    logger.info('saving image shape: %s with %s' %
      (str(image.shape), str(label)))
    self.entries.append((buffer(image),
      image.shape[1], image.shape[0], image.shape[2], str(label),))

  def stop(self):
    self.running = False
    logger.info('stopping saver...')


class Display(object):
  def __init__(self, output_db, table_name, display_size,
      camera_id, params, adjust_key, target_coord, min_area):
    self.display_size = display_size
    self.adjust_key = adjust_key
    self.target_coord = np.zeros(shape=[4, 2], dtype=np.float32)
    for i, coord in enumerate(target_coord):
      self.target_coord[i, :] = np.array(coord)
    self.source_coord = np.float32(
      [[0.0, 0.0], [display_size[0] / 2, 0],
       [0.0, display_size[1] / 2],
       [display_size[0] / 2, display_size[1] / 2]])
    self.p_maxtrix = cv2.getPerspectiveTransform(self.target_coord,
      self.source_coord)

    self.min_area = min_area

    self.param = params
    self.running = True
    self.camera = cv2.VideoCapture(camera_id)

    self.saver = Saver(output_db, table_name)
    self.saver.start()

  def get_frame(self):
    _, raw_frame = self.camera.read()
    return raw_frame

  def start(self):
    display = np.zeros(shape=[self.display_size[1],
      self.display_size[0], 3], dtype=np.uint8)
    w = self.display_size[0]
    h = self.display_size[1]
    while self.running:
      raw_frame = self.get_frame()
      raw_frame = cv2.resize(raw_frame, (w / 2, h / 2))
      display[:h/2, :w/2, :] = raw_frame
      platform = cv2.warpPerspective(raw_frame,
        self.p_maxtrix, (w/2, h/2))

      # compute hsv
      _, filtered, roi_output, roi = threshold_characters(
        platform, self.param, self.min_area)
      display[h/2:, :w/2, :] = filtered
      display[:h/2, w/2:, :] = roi_output

      cv2.imshow('Display', display)
      key = cv2.waitKey(20)
      if key in [ord('q'), ord('Q')]:
        logger.info('closing display...')
        break
      elif key in [ord('c'), ord('C')]:
        display[h/2:, w/2:, :] = roi_output
      elif key in [ord('s'), ord('S')]:
        area = display[h/2:, w/2:, :]
        if np.sum(area[area > 0]) > 0:
          logger.info('saving entry...')
          self.saver.save_entry(platform, roi)
          display[h/2:, w/2:, :] = 0
        else:
          logger.warning('not captured.')
      elif key in [ord('y'), ord('Y')]:
        self.param[self.adjust_key]['HHigh'] -= 1
        logger.info("HHigh: %d" % self.param[self.adjust_key]['HHigh'])
      elif key in [ord('6'), ord('^')]:
        self.param[self.adjust_key]['HHigh'] += 1
        logger.info("HHigh: %d" % self.param[self.adjust_key]['HHigh'])
      elif key in [ord('h'), ord('H')]:
        self.param[self.adjust_key]['HLow'] += 1
        logger.info("HLow: %d" % self.param[self.adjust_key]['HLow'])
      elif key in [ord('n'), ord('N')]:
        self.param[self.adjust_key]['HLow'] -= 1
        logger.info("HLow: %d" % self.param[self.adjust_key]['HLow'])
      elif key in [ord('7'), ord('&')]:
        self.param[self.adjust_key]['SHigh'] += 1
        logger.info("SHigh: %d" % self.param[self.adjust_key]['SHigh'])
      elif key in [ord('u'), ord('U')]:
        self.param[self.adjust_key]['SHigh'] -= 1
        logger.info("SHigh: %d" % self.param[self.adjust_key]['SHigh'])
      elif key in [ord('j'), ord('J')]:
        self.param[self.adjust_key]['SLow'] += 1
        logger.info("SLow: %d" % self.param[self.adjust_key]['SLow'])
      elif key in [ord('m'), ord('M')]:
        self.param[self.adjust_key]['SLow'] -= 1
        logger.info("SLow: %d" % self.param[self.adjust_key]['SLow'])
      elif key in [ord('8'), ord('*')]:
        self.param[self.adjust_key]['VHigh'] += 1
        logger.info("VHigh: %d" % self.param[self.adjust_key]['VHigh'])
      elif key in [ord('i'), ord('I')]:
        self.param[self.adjust_key]['VHigh'] -= 1
        logger.info("VHigh: %d" % self.param[self.adjust_key]['VHigh'])
      elif key in [ord('k'), ord('K')]:
        self.param[self.adjust_key]['VLow'] += 1
        logger.info("VLow: %d" % self.param[self.adjust_key]['VLow'])
      elif key in [ord(','), ord('<')]:
        self.param[self.adjust_key]['VLow'] -= 1
        logger.info("VLow: %d" % self.param[self.adjust_key]['VLow'])
    logger.info('stopping saver...')
    self.saver.running = False
    cv2.destroyAllWindows()


def main():
  default_width, default_height = 1024, 768
  parser = ArgumentParser()
  parser.add_argument('--output-db', dest='output_db',
    default='roi.sqlite3', help='output sqlite3 database')
  parser.add_argument('--table-name', dest='table_name',
    default='roi', help='output table name')
  parser.add_argument('--param', dest='param',
    help='param json')
  parser.add_argument('--display-width', dest='display_width',
    default=default_width, type=int, help='single display width')
  parser.add_argument('--display-height', dest='display_height',
    default=default_height, type=int, help='single display height')
  parser.add_argument('--min-area', dest='min_area',
    default=150, type=int, help='min area to be consider as roi')

  parser.add_argument('--top-left', dest='top_left',
    default='0,0', type=str,
    help='top left coordinate for perspective transform')
  parser.add_argument('--top-right', dest='top_right',
    default='%d,0' % default_width, type=str,
    help='top right coordinate for perspective transform')
  parser.add_argument('--bot-left', dest='bot_left',
    default='0,%d' % default_height, type=str,
    help='bottom left coordinate for perspective transform')
  parser.add_argument('--bot-right', dest='bot_right',
    default='%d,%d' % (default_width, default_height), type=str,
    help='bottom right coordinate for perspective transform')

  parser.add_argument('--adjust-key', dest='adjust_key',
    default='wood', type=str,
    help='filter hsv value to adjust')

  parser.add_argument('--camera-id', dest='camera_id',
    default=0, type=int, help='camera id for capture images')
  args = parser.parse_args()

  def to_coord(coord_str):
    cs = coord_str.split(',')
    return [float(c) for c in cs]

  if os.path.isfile(args.param):
    with open(args.param, 'r') as f:
      param = json.loads(f.read())
      display_size = (args.display_width, args.display_height)
      target_coord = [
        to_coord(args.top_left),
        to_coord(args.top_right),
        to_coord(args.bot_left),
        to_coord(args.bot_right),
      ]

      display = Display(args.output_db,
        args.table_name,
        display_size,
        args.camera_id,
        param,
        args.adjust_key,
        target_coord,
        args.min_area)
      display.start()
  else:
    logger.info('%s parameter does not exists' % (args.param))


if __name__ == '__main__':
  main()

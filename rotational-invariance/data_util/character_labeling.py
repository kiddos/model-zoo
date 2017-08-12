import sqlite3
import os
import logging
import json
import time
import Tkinter as tk
import tkMessageBox as tkm
import numpy as np
import ast
import cv2
import threading
import math
from PIL import Image, ImageTk
from collections import deque
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_characters(json_file):
  if os.path.isfile(json_file):
    with open(json_file, 'r') as f:
      return json.load(f)
  else:
    return None


def load_roi_data(input_db, table_name):
  if os.path.join(input_db):
    logger.info('loading %s...' % (input_db))
    connection = sqlite3.connect(input_db)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM %s;""" % (table_name))
    raw_data = cursor.fetchall()
    connection.close()
    return raw_data
  else:
    logger.info('%s not found' % (input_db))


class CharacterSaver(object):
  def __init__(self, output_db, table_name):
    self.output_db = output_db
    self.table_name = table_name
    self.running = False
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
      char TEXT NOT NULL,
      rotation FLOAT NOT NULL);""" % (self.table_name))

    self.running = True
    while self.running:
      if len(self.entries) > 0:
        entry = self.entries.popleft()
        cursor.execute("""INSERT INTO %s(
          image, imageWidth, imageHeight, imageChannel, char, rotation)
          VALUES(?, ?, ?, ?, ?, ?)""" %
          (self.table_name), entry)
        logger.info('inserting (%d, %d, %d) image: %s with %s rad' %
          (entry[1], entry[2], entry[3], entry[4], entry[5]))
      time.sleep(0.5)
    logger.info('flusing data...')
    while len(self.entries) > 0:
      cursor.execute("""INSERT INTO %s(
        image, imageWidth, imageHeight, imageChannel, char, rotation)
        VALUES(?, ?, ?, ?, ?, ?)""" %
        (self.table_name), entry)
    connection.commit()
    connection.close()

  def add_entry(self, image, char, rotation):
    self.entries.append((buffer(image), image.shape[1], image.shape[0],
      image.shape[2], char, rotation))

  def start(self):
    self.task = threading.Thread(target=self.run)
    self.task.start()


class CharacterLabelUI(object):
  def __init__(self, raw_data, characters, output_db, table_name, padding):
    self.saver = CharacterSaver(output_db, table_name)
    self.saver.start()
    self.characters = characters

    self.padding = padding
    self.images = []
    for entry in raw_data:
      img = np.frombuffer(entry[1], np.uint8).reshape([
        entry[3], entry[2], entry[4]])
      roi = ast.literal_eval(entry[5])
      for rect in roi:
        x = np.max([rect[0] - self.padding, 0])
        y = np.max([rect[1] - self.padding, 0])
        w = np.min([rect[2] + self.padding * 2, img.shape[1]])
        h = np.min([rect[3] + self.padding * 2, img.shape[0]])
        roi_image = np.copy(img[y:y+h, x:x+w, :])
        self.images.append(cv2.resize(
          cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB), (200, 200)))
    # ui
    self.root = tk.Tk()
    self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    self.root.bind('<Key>', self.key_event)
    self.canvas = tk.Canvas(self.root, width=600, height=600)
    self.canvas.bind('<ButtonPress-1>', self.on_press)
    self.canvas.pack(side=tk.LEFT)

    self.info_pane = tk.PanedWindow(self.root)
    self.info_pane.pack(side=tk.RIGHT)
    self.rotation_var = tk.StringVar(self.info_pane)
    self.rotation_var.set('0')
    self.rotation_label = tk.Label(self.info_pane,
      textvariable=self.rotation_var)
    self.rotation_label.pack(side=tk.TOP)
    self.character_var = tk.StringVar(self.info_pane)
    self.character_var.set('')
    self.character_label = tk.Label(self.info_pane,
      textvariable=self.character_var)
    self.character_label.pack(side=tk.TOP)

    self.sx, self.sy = 300, 0
    self.index = 0
    self._render()

  def key_event(self, event):
    char_val = 0
    try:
      char_val = ord(event.char)
    except:
      char_val = -1
    if char_val == 13:
      result = tkm.askokcancel('Delete',
        'Are you sure you want to delete this entry')
      if result:
        self.save()
    elif char_val == ord('q'):
      self.on_close()
    elif unicode(event.char.upper()) in self.characters['characters']:
      self.character_var.set(event.char.upper())
    elif event.char == ' ':
      self.load_next()
    else:
      self.character_var.set('others')

  def load_image(self):
    image = self.images[self.index]
    img = Image.fromarray(image)
    self.tk_image = ImageTk.PhotoImage(img)
    self.canvas.create_image(
      300, 300, image=self.tk_image)

  def on_press(self, event):
    self.mouse_press = True
    dx = event.x - 300
    dy = event.y - 300
    rotation = math.atan2(dy, dx)
    self.rotation_var.set(str(rotation + math.pi / 2))

    self.sx = 300 + 300 * math.cos(rotation)
    self.sy = 300 + 300 * math.sin(rotation)
    self._render()

  def save(self):
    rotation = float(self.rotation_var.get())
    character = self.character_var.get()
    if character != '':
      self.saver.add_entry(self.images[self.index], character, rotation)
      self.load_next()
    else:
      logger.warning('character not labeled')

  def load_next(self):
    self.index += 1
    if self.index == len(self.images):
      self.on_close()
    else:
      self._render()

  def run(self):
    self.root.mainloop()

  def on_close(self):
    self.saver.running = False
    self.root.destroy()

  def _render(self):
    self.canvas.create_rectangle(0, 0, 600, 600, fill='white')
    self.load_image()
    self.canvas.create_line(self.sx, self.sy, 300, 300, fill='green')


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-db', dest='input_db',
    default='raw_roi.sqlite3', help='input db to label characters')
  parser.add_argument('--output-db', dest='output_db',
    default='characters.sqlite3', help='output db label characters')
  parser.add_argument('--table-name', dest='table_name',
    default='roi', help='output table name')
  parser.add_argument('--padding', dest='padding',
    default=10, type=int, help='output table name')
  args = parser.parse_args()

  data = load_roi_data(args.input_db, args.table_name)
  character = get_characters('character.json')
  ui = CharacterLabelUI(data, character, args.output_db, args.table_name,
      args.padding)
  ui.run()




if __name__ == '__main__':
  main()

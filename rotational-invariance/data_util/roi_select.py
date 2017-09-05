import os
import sqlite3
import logging
import time
import Tkinter as tk
import tkMessageBox as tkm
import numpy as np
import cv2
import threading
from PIL import Image, ImageTk
from collections import deque
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(input_db, table_name):
  old_data = None
  if os.path.isfile(input_db):
    logger.info('loading %s...' % (input_db))
    connection = sqlite3.connect(input_db)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM %s;""" % (
      table_name))
    old_data = cursor.fetchall()
    return old_data
    connection.close()
  else:
    logger.info('%s is not found' % (input_db))


class Deleter(object):
  def __init__(self, db_name, table_name):
    self.table_name = table_name
    self.db_name = db_name
    self.entries_to_delete = deque()

  def run(self):
    self.connection = sqlite3.connect(self.db_name)
    self.cursor = self.connection.cursor()
    self.running = True
    while self.running:
      if len(self.entries_to_delete) > 0:
        entry = self.entries_to_delete.popleft()
        self.cursor.execute("""DELETE FROM %s where id = %d""" %
          (self.table_name, entry[0]))
      time.sleep(0.5)
    logger.info('delete rest...')
    while len(self.entries_to_delete) > 0:
      entry = self.entries_to_delete.popleft()
      self.cursor.execute("""DELETE FROM %s where id = %d""" %
        (self.table_name, entry[0]))
    logger.info('closing db...')
    self.connection.commit()
    self.connection.close()

  def delete_entry(self, entry):
    self.entries_to_delete.append(entry)

  def start(self):
    self.task = threading.Thread(target=self.run)
    self.task.start()


class EditorUI(object):
  def __init__(self, data, db_name, table_name):
    self.deleter = Deleter(db_name, table_name)
    self.deleter.start()
    self.data = data
    self.root = tk.Tk()
    self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    self.root.bind('<Key>', self.key_event)
    self.width, self.height = data[0][2], data[0][3]
    self.canvas = tk.Canvas(self.root,
      width=data[0][2], height=data[0][3])
    self.canvas.pack()

    self.index = 0
    self.load_image()

  def load_image(self):
    d = self.data[self.index]
    image = cv2.cvtColor(np.frombuffer(d[1],
      dtype=np.uint8).reshape([d[3], d[2], d[4]]),
      cv2.COLOR_BGR2RGB)
    grid = np.frombuffer(d[5], np.float32).reshape(
      [d[7], d[6], d[8]])
    roi_grid = grid[:, :, 0] > 0
    roi = grid[roi_grid, 1:]
    for i in range(len(roi)):
      cx = roi[i, 0] * d[2]
      cy = roi[i, 1] * d[3]
      w = roi[i, 2] * d[2]
      h = roi[i, 3] * d[3]
      p1 = (int(cx - w / 2), int(cy - h / 2))
      p2 = (int(cx + w / 2), int(cy + h / 2))
      cv2.rectangle(image, p1, p2, (0, 255, 0), 2)

    img = Image.fromarray(image)
    self.tk_image = ImageTk.PhotoImage(img)
    self.canvas.create_image(
      self.width / 2, self.height / 2, image=self.tk_image)

  def key_event(self, event):
    if event.char == 'n':
      self.load_next()
    elif event.char == 'd':
      self.delete()

  def load_next(self):
    self.index += 1
    if self.index == len(self.data):
      logger.info('done labeling, closing...')
      self.on_close()
    else:
      self.load_image()

  def delete(self):
    result = tkm.askokcancel('Delete',
      'Are you sure you want to delete this entry')
    if result:
      self.deleter.delete_entry(self.data[self.index])
    self.load_next()

  def run(self):
    self.root.mainloop()

  def on_close(self):
    self.deleter.running = False
    self.root.destroy()


def main():
  parser = ArgumentParser()
  parser.add_argument('--db', dest='db', default='roi.sqlite3',
    type=str, help='sqlite3 database to view and change')
  parser.add_argument('--table-name', dest='table_name',
    default='roi', help='output table name')
  args = parser.parse_args()

  data = load_data(args.db, args.table_name)
  ui = EditorUI(data, args.db, args.table_name)
  ui.run()


if __name__ == '__main__':
  main()

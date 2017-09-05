import os
import logging
import Tkinter as tk
import numpy as np
import sqlite3
import time
import threading
import ast
from collections import deque
from PIL import Image, ImageTk
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Updater(object):
  def __init__(self, db_name, table_name):
    self.table_name = table_name
    self.db_name = db_name

  def get_all_entries(self):
    self.connection = sqlite3.connect(self.db_name)
    self.cursor = self.connection.cursor()
    self.cursor.execute("""SELECT * FROM %s;""" % (self.table_name))
    data = self.cursor.fetchall()
    self.connection.close()
    return data

  def run(self):
    self.connection = sqlite3.connect(self.db_name)
    self.cursor = self.connection.cursor()
    self.entry = deque()
    self.running = True
    while self.running:
      if len(self.entry) > 0:
        e = self.entry.popleft()
        logger.info('updating entry %d. %s: %s' % (e[0], e[1].shape, str(e[2])))
        self.cursor.execute("""UPDATE %s SET label = \"%s\" WHERE id = %d;""" %
          (self.table_name, str(e[2]), e[0]))
      time.sleep(0.5)

    logger.info('flusing data...')
    while len(self.entry) > 0:
      e = self.entry.popleft()
      self.cursor.execute(("""INSERT INTO %s(image, label) VALUES(?, ?);""" %
        (self.table_name)), (buffer(e[0].reshape(-1)), str(e[1]),))
    self.connection.commit()
    self.connection.close()

  def update_async(self, id, image, label):
    logger.info('%d. %s:%s' % (id, str(image.shape), str(label)))
    self.entry.append([id, image, str(label)])

  def start(self):
    self.task = threading.Thread(target=self.run)
    self.task.start()


class UpdateLabelUI(object):
  def __init__(self, db_name, table_name,
      image_width=640, image_height=480):
    self.updater = Updater(db_name, table_name)
    raw_entries = self.updater.get_all_entries()
    self.images = [(id, np.frombuffer(image_buffer, np.uint8).reshape([height, width, 3]),
      label) for id, image_buffer, width, height, label in raw_entries]
    self.image_index = 0
    self.rois = ast.literal_eval(self.images[self.image_index][2])
    self.updater.start()

    self.width = image_width
    self.height = image_height

    self.root = tk.Tk()
    self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    self.canvas = tk.Canvas(self.root,
      width=image_width, height=image_height)
    self.canvas.bind('<ButtonPress-1>', self.on_press)
    self.canvas.bind('<B1-Motion>', self.on_move)
    self.canvas.bind('<ButtonRelease-1>', self.on_release)
    self.canvas.pack(side=tk.LEFT)

    self.labeling_panel = tk.PanedWindow(self.root)
    self.save_button = tk.Button(self.labeling_panel, text='Save',
      command=self.save_image)
    self.save_button.pack(side=tk.TOP)

    self.skip_button = tk.Button(self.labeling_panel, text='Skip',
      command=self.skip_image)
    self.skip_button.pack(side=tk.TOP)

    self.remove_last_button = tk.Button(self.labeling_panel, text='Remove Last',
      command=self.remove_last)
    self.remove_last_button.pack()

    block_choise = [
      ['H', 'H'],
      ['I', 'I'],
      ['W', 'W'],
      ['N', 'N'],
      ['R', 'R'],
      ['O', 'O'],
      ['B', 'B'],
      ['T', 'T'],
      ['2', '2'],
      ['0', '0'],
      ['1', '1'],
      ['7', '7'],
      ['others', 'others']
    ]
    self.choise_var = tk.StringVar(self.labeling_panel, value=block_choise[0][0])
    self.choise = []
    for text, mode in block_choise:
      b = tk.Radiobutton(self.labeling_panel,
        text=text,
        value=mode,
        variable=self.choise_var,
        indicatoron=1)
      b.deselect()
      b.pack(anchor=tk.W)
      self.choise.append(b)
    self.labeling_panel.pack(side=tk.RIGHT)

    self.mouse_press = False
    self.render()


  def on_close(self):
    self.updater.running = False
    self.root.destroy()

  def load_image(self):
    image = self.images[self.image_index][1]
    img = Image.fromarray(image)
    self.tk_image = ImageTk.PhotoImage(img)
    self.canvas.create_image(
      self.width / 2, self.height / 2, image=self.tk_image)

  def on_press(self, event):
    self.mouse_press = True
    self.sx = event.x
    self.sy = event.y
    self.cx = event.x
    self.cy = event.y
    self.render()

  def on_move(self, event):
    self.cx = event.x
    self.cy = event.y
    self.render()

  def on_release(self, event):
    self.mouse_press = False
    self.cx = event.x
    self.cy = event.y
    if self.choise_var.get() != '':
      self.rois.append([self.sx, self.sy, self.cx, self.cy, self.choise_var.get()])
    self.render()

  def render(self):
    text_padding = 10
    self.load_image()
    for roi in self.rois:
      self.canvas.create_rectangle(roi[0], roi[1],
        roi[2], roi[3], width=2, outline='green')
      self.canvas.create_text(roi[0], roi[1] - text_padding, text=roi[4],
        fill='green', width=2)
    # draw temp
    if self.mouse_press:
      self.canvas.create_rectangle(self.sx, self.sy, self.cx, self.cy, width=2,
        outline='red')
      self.canvas.create_text(self.sx, self.sy - text_padding,
        text=self.choise_var.get(),
        width=2, fill='red')

  def remove_last(self):
    if len(self.rois) > 0:
      self.rois = self.rois[:-1]
    self.render()

  def load_next(self):
    self.image_index += 1
    if self.image_index == len(self.images):
      logger.info('done labeling, closing...')
      self.on_close()
    else:
      self.load_image()
      self.rois = ast.literal_eval(self.images[self.image_index][2])
      self.render()

  def skip_image(self):
    self.load_next()

  def save_image(self):
    entry = self.images[self.image_index]
    self.updater.update_async(entry[0], entry[1], self.rois)
    self.skip_image()

  def start(self):
    self.root.mainloop()


def main():
  parser = ArgumentParser()
  parser.add_argument('--db-name', dest='db_name',
    default='duckymomo.sqlite3', help='output database name',)
  parser.add_argument('--table-name', dest='table_name',
    default='blocks', help='output database name',)
  parser.add_argument('--folder', dest='folder',
    default='duckymomo', help='input image folder')
  args = parser.parse_args()

  ui = UpdateLabelUI(args.db_name, args.table_name)
  ui.start()


if __name__ == '__main__':
  main()

from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import sqlite3
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

import background_data_util
import font_data_util


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
COLORS = [
    (255, 255, 255), (255, 0, 0), (0, 255, 0),
    (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def load(font_size=48):
  images = background_data_util.load()
  images = [Image.open(path) for path in images]
  fonts = font_data_util.load()
  fonts = [ImageFont.truetype(path, font_size) for path in fonts]
  return images, fonts


def create_font_image(images, fonts, image_index, font_index,
    text='A', rotation=0, padding=12, output_size=64, color=(255, 0, 0)):
  assert image_index >= 0 and image_index < len(images)
  assert font_index >= 0 and font_index < len(fonts)

  img = images[image_index].copy()
  font = fonts[font_index]

  assert output_size < img.size[0] and output_size < img.size[1]

  txt = Image.new('L', (output_size, output_size))
  txt_draw = ImageDraw.Draw(txt)
  w, h = txt_draw.textsize(text, font=font)
  txt_draw.text(((output_size - w)/2, (output_size - h - padding)/2),
      text, font=font, fill=255)
  rotated = txt.rotate(rotation)

  x = random.randrange(0, img.size[0] - output_size)
  y = random.randrange(0, img.size[1] - output_size)
  img = img.crop((x, y, x + output_size, y + output_size))
  img.paste(ImageOps.colorize(rotated, (0, 0, 0), color), (0, 0),
      rotated)
  return img


def create_db(db_name, image_count, characters, colors, image_size=64):
  connection = sqlite3.connect(db_name)
  cursor = connection.cursor()
  cursor.execute("""CREATE TABLE IF NOT EXISTS characters(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      data BLOB NOT NULL,
      width INTEGER,
      height INTEGER,
      rotation REAL,
      char TEXT
  );""")

  images, fonts = load()
  logger.info('inserting data into %s...' % (db_name))
  for i in range(image_count):
    image_index = random.randrange(0, len(images))
    font_index = random.randrange(0, len(fonts))
    rotation = random.randrange(0, 359)
    text = characters[random.randrange(0, len(characters))]
    color = colors[random.randrange(0, len(colors))]
    img = create_font_image(images, fonts, image_index, font_index, text=text,
        rotation=rotation, output_size=image_size, color=color)
    if rotation > 180:
      rotation -= 360
    cursor.execute("""INSERT INTO characters(
        data, width, height, rotation, char) values (?, ?, ?, ?, ?);""",
      (buffer(np.array(img).tobytes()),
          image_size, image_size, rotation * math.pi / 180.0, text,))
  connection.commit()


def load_data(db_name, characters):
  connection = sqlite3.connect(db_name)
  cursor = connection.cursor()
  cursor.execute("""SELECT * FROM characters;""")
  entries = cursor.fetchall()

  data = []
  label = []
  num_label = len(characters)
  logger.info('loading data from %s...' % (db_name))
  for entry in entries:
    w = entry[2]
    h = entry[3]
    data.append(np.array(entry[1]).reshape(h, w, 3))
    index = next(i for i, c in enumerate(characters) if c == entry[5])
    label.append([1 if l == index else 0 for l in range(num_label)] +
        [entry[4]])
  return np.array(data), np.array(label)


def main():
  parser = OptionParser()
  parser.add_option('-n', '--num_images', dest='num_images', default=10000,
      help='number of images to add to database')
  parser.add_option('-o', '--output', dest='output_db',
      default='characters_train.sqlite',
      help='output sqlite database path')
  option, args = parser.parse_args()

  create_db(option.output_db, int(option.num_images), CHARACTERS, COLORS)
  data, label = load_data(option.output_db, CHARACTERS)
  print('data shape: %s' % (str(data.shape)))

  index = random.randrange(len(data))
  plt.imshow(data[index])
  print(label[index, :])
  plt.show()


if __name__ == '__main__':
  main()

from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import sqlite3
import logging
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from argparse import ArgumentParser

import background_data_util
import font_data_util


logging.basicConfig()
logger = logging.getLogger('yolo_data_util')
logger.setLevel(logging.INFO)


def load():
  images = background_data_util.load()
  images = [Image.open(path) for path in images]
  fonts_path = font_data_util.load()
  return images, fonts_path


def draw_image(background_image, image_width, image_height,
    character_txts, colors, coordinates):
  """
  draw the character text on background image
  """
  assert len(character_txts) == len(colors)
  assert len(character_txts) == len(coordinates)

  img = cv2.resize(np.array(background_image.copy()),
      (image_width, image_height))
  img = Image.fromarray(img)

  char_count = len(character_txts)
  for i in range(char_count):
    img.paste(ImageOps.colorize(character_txts[i],
      (0, 0, 0), colors[i]),
      coordinates[i],
      character_txts[i])
  return img


def occupied(coordinates, coord, output_width, output_height):
  for c in coordinates:
    if int(c[0] / output_width) == \
        int(coord[0] / output_width) and \
        int(c[1] / output_height) == \
        int(coord[1] / output_height):
      return True
  return False


def create_random_image(background_images,
    input_width, input_height,
    output_width, output_height,
    font_paths, selectable_colors, selectable_chars,
    max_select_count=3, padding=12, min_font_size=12,
    bounding_box=1):
  """
  create a image with draws random characters
  the characters output grid will not overlap each other
  which will only output 1 bounding box for each grid cell
  """
  background_image = background_images[
    random.randrange(0, len(background_images))]
  select_count = random.randrange(0, max_select_count + 1)
  character_txts = []
  char_indexes = []
  colors = []
  coordinates = []
  sizes = []
  rotations = []
  xys = []
  for i in range(select_count):
    # random colors
    color = selectable_colors[
      random.randrange(0, len(selectable_colors))]
    colors.append(color)
    # random font
    max_font_size = min(input_width, input_height) / select_count
    font_size = random.randrange(min_font_size, max_font_size)
    font = ImageFont.truetype(font_paths[
      random.randrange(0, len(font_paths))], font_size)
    # random character
    ci = random.randrange(0, len(selectable_chars))
    char = selectable_chars[ci]
    char_indexes.append(ci)
    # create character image
    txt = Image.new('L', (font_size + padding, font_size + padding))
    txt_draw = ImageDraw.Draw(txt)
    w, h = txt_draw.textsize(char, font=font)
    txt_draw.text(((font_size + padding - w)/2, (font_size + padding - h)/2),
        char, font=font, fill=255)
    sizes.append((w, h))
    # random rotation
    rotation = np.random.uniform(low=-180, high=180)
    rotations.append(rotation)
    rotated = txt.rotate(rotation)
    character_txts.append(rotated)
    # random coordinate
    coord = (
      random.randrange(-font_size / 2, input_width - font_size / 2 + 1),
      random.randrange(-font_size / 2, input_height - font_size / 2 + 1)
    )
    while occupied(coordinates, coord, output_width, output_height):
      coord = (
        random.randrange(-font_size / 2, input_width - font_size / 2 + 1),
        random.randrange(-font_size / 2, input_height - font_size / 2 + 1)
      )
    coordinates.append(coord)
    xys.append((coord[0] + float(font_size + padding) / 2,
      coord[1] + float(font_size + padding) / 2))
  # draw the image
  character_image = draw_image(background_image,
    input_width, input_height, character_txts, colors, coordinates)
  # prepare output grid
  num_classes = len(selectable_chars)
  grid = np.zeros(shape=[output_height, output_width,
    bounding_box * 6 + num_classes], dtype=np.float32)
  for i, c in enumerate(xys):
    x = int(np.round(c[0] / input_width * output_width))
    y = int(np.round(c[1] / input_height * output_height))
    x = max(min(output_width - 1, x), 0)
    y = max(min(output_height - 1, y), 0)
    grid[y, x, 0] = 1  # indicator
    grid[y, x, 1] = c[0] / input_width  # x
    grid[y, x, 2] = c[1] / input_height  # y
    grid[y, x, 3] = float(sizes[i][0]) / input_width  # width
    grid[y, x, 4] = float(sizes[i][1]) / input_height  # height
    grid[y, x, 5] = -rotations[i] / 180.0 * np.pi  # rotation
    grid[y, x, 6 + char_indexes[i]] = 1.0  # class
  return np.array(character_image, np.uint8), grid


def create_yolo_db(db_name, table_name,
    background_images,
    input_width, input_height,
    output_width, output_height,
    font_paths, selectable_colors, selectable_chars,
    max_select_count=3, entry_count=10000):
  """
  create the yolo data with output label grid + rotation
  """
  create_character_table = os.path.isfile(db_name)

  connection = sqlite3.connect(db_name)
  cursor = connection.cursor()
  # create character table if db is first time created
  if create_character_table:
    logger.info('creating character table')
    cursor.execute("""CREATE TABLE IF NOT EXISTS characters(
      id INTEGER PRIMARY KEY,
      character TEXT);""")
    for i in range(len(selectable_chars)):
      cursor.execute("""INSERT INTO characters(character)
        VALUES(?);""", (selectable_chars[i],))

  # create data table
  logger.info('creating yolo data table')
  cursor.execute("""CREATE TABLE IF NOT EXISTS %s(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB NOT NULL,
    imageWidth INTEGER NOT NULL,
    imageHeight INTEGER NOT NULL,
    imageChannel INTEGER NOT NULL,
    grid BLOB NOT NULL,
    gridWidth INTEGER NOT NULL,
    gridHeight INTEGER NOT NULL,
    gridChannel INTEGER NOT NULL);""" % (table_name))

  for i in range(entry_count):
    img, label = create_random_image(background_images,
      input_width, input_height,
      output_width, output_height,
      font_paths, selectable_colors, selectable_chars, max_select_count)
    cursor.execute(("""INSERT INTO %s(
      image, imageWidth, imageHeight, imageChannel,
      grid, gridWidth, gridHeight, gridChannel)
      VALUES(?, ?, ?, ?, ?, ?, ?, ?);""" % (table_name)),
        (buffer(img), input_width, input_height, 3,
          buffer(label), output_width, output_height, label.shape[2],))
    if i % 100 == 0 and i != 0:
      logger.info('progress: %d/%d' % (i, entry_count))
  logger.info('commit...')
  connection.commit()
  connection.close()


def load_yolo_data(db_name, table_name):
  connection = sqlite3.connect(db_name)
  cursor = connection.cursor()
  cursor.execute("""SELECT image, imageWidth, imageHeight, imageChannel,
    grid, gridWidth, gridHeight, gridChannel FROM %s;""" % (table_name))
  raw_data = cursor.fetchall()

  images = []
  labels = []
  for entry in raw_data:
    images.append(np.frombuffer(
      entry[0], np.uint8).reshape([entry[2], entry[1], entry[3]]))
    labels.append(np.frombuffer(
      entry[4], np.float32).reshape([entry[6], entry[5], entry[7]]))
  connection.close()
  return np.array(images, np.uint8), np.array(labels)

def create_image(image_width, image_height,
    images, fonts_path, image_index, font_index,
    text='A', padding=12, color=(255, 0, 0),
    min_font_size=10, max_font_size=110):
  assert image_index >= 0 and image_index < len(images)
  assert font_index >= 0 and font_index < len(fonts_path)

  img = cv2.resize(np.array(images[image_index].copy()),
      (image_width, image_height))
  img = Image.fromarray(img)
  font_path = fonts_path[font_index]

  font_size = np.round(np.random.uniform(min_font_size,
      max_font_size)).astype(np.int32)
  font = ImageFont.truetype(font_path, font_size)
  assert font_size < img.size[0] and font_size < img.size[1]

  txt = Image.new('L', (font_size + padding, font_size + padding))
  txt_draw = ImageDraw.Draw(txt)
  w, h = txt_draw.textsize(text, font=font)
  txt_draw.text(((font_size - w)/2, (font_size - h)/2),
      text, font=font, fill=255)
  rotation = np.random.uniform(low=0, high=360)
  rotated = txt.rotate(rotation)

  x = random.randrange(0, img.size[0] - font_size - 1)
  y = random.randrange(0, img.size[1] - font_size - 1)
  img.paste(ImageOps.colorize(rotated, (0, 0, 0), color), (x, y),
      rotated)
  return img, rotation, x, y, font_size + padding, font_size + padding


def create_db(db_name, image_width, image_height,
    image_count, characters, colors):
  connection = sqlite3.connect(db_name)
  cursor = connection.cursor()
  cursor.execute("""CREATE TABLE IF NOT EXISTS characters(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      data BLOB NOT NULL,
      image_width INTEGER,
      image_height INTEGER,
      x INTEGER,
      y INTEGER,
      w INTEGER,
      h INTEGER,
      rotation REAL,
      char TEXT);""")

  images, fonts_path = load()
  logger.info('inserting data into %s...' % (db_name))
  for i in range(image_count):
    image_index = random.randrange(0, len(images))
    font_index = random.randrange(0, len(fonts_path))
    text = characters[random.randrange(0, len(characters))]
    color = colors[random.randrange(0, len(colors))]
    img, r, x, y, w, h = create_image(image_width, image_height,
      images, fonts_path, image_index, font_index, text=text,
      color=color)
    if r > 180: r -= 360
    assert y >= 0 and x >= 0 and w > 0 and h > 0
    cursor.execute("""INSERT INTO characters(
      data, image_width, image_height, x, y, w, h, rotation, char)
      values (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
      (buffer(np.array(img).tobytes()), image_width, image_height,
        x, y, w, h, r* math.pi / 180.0, text,))
  connection.commit()
  connection.close()


def load_data(db_name, characters,
    output_width, output_height, output_classes, bounding_box=1):
  connection = sqlite3.connect(db_name)
  cursor = connection.cursor()
  cursor.execute("""SELECT * FROM characters;""")
  entries = cursor.fetchall()

  images = []
  coordinates = []
  dimensions = []
  classes = []
  logger.info('loading data from %s...' % (db_name))
  for entry in entries:
    # image
    iw = entry[2]
    ih = entry[3]
    width_scale = iw / output_width
    height_scale = ih / output_height
    images.append(np.array(entry[1]).reshape(ih, iw, 3))

    x = entry[4]
    y = entry[5]
    w = entry[6]
    h = entry[7]
    #  print(x, y, w, h, iw, ih, width_scale, height_scale)
    cx = np.floor((x + w / 2) / width_scale).astype(np.int32)
    cy = np.floor((y + h / 2) / height_scale).astype(np.int32)
    #  print(cx, cy)
    x = x * 1.0 / iw
    y = y * 1.0 / ih
    w = w * 1.0 / iw
    h = h * 1.0 / ih
    # coordinate
    coord = np.zeros(
      shape=[output_height, output_width, bounding_box * 2 * output_classes])
    coord[cy, cx, 0] = x
    coord[cy, cx, 1] = y
    coordinates.append(coord)
    # dimension
    dim = np.zeros(
      shape=[output_height, output_width, bounding_box * 2 * output_classes])
    dim[cy, cx, 0] = w
    dim[cy, cx, 1] = h
    dimensions.append(dim)
    # target class
    l = np.zeros(shape=[output_height, output_width, output_classes])
    l[cy, cx, 0] = 1.0
    classes.append(l)
  return np.array(images), np.array(coordinates), np.array(dimensions), \
    np.array(classes)


def main():
  parser = ArgumentParser()
  parser.add_argument('--num-images', dest='num_images',
    default=10000, type=int, help='number of images to add to database')
  parser.add_argument('--input-width', dest='input_width',
    default=160, type=int, help='input image width')
  parser.add_argument('--input-height', dest='input_height',
    default=120, type=int, help='input image height')
  parser.add_argument('--output-width', dest='output_width',
    default=16, type=int, help='output image width')
  parser.add_argument('--output-height', dest='output_height',
    default=12, type=int, help='output image height')
  parser.add_argument('--characters', dest='characters',
    default='A,B,C,D,E,F,G,H,J,K', type=str,
    help='characters to create image from')

  parser.add_argument('-o', dest='output_db',
    default='yolo.sqlite3', type=str, help='output sqlite database path')
  parser.add_argument('--table-name', dest='table_name',
    default='yolo', type=str, help='output sqlite data table name')
  parser.add_argument('--max-count', dest='max_select_count',
    default=6, type=int, help='max number of characters in 1 image')
  args = parser.parse_args()

  chars = args.characters.split(',')
  selectable_colors = [
      (255, 255, 255), (255, 0, 0), (0, 255, 0),
      (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

  images = background_data_util.load()
  background_images = [Image.open(path) for path in images]
  font_paths = font_data_util.load()
  create_yolo_db(args.output_db, args.table_name,
    background_images,
    args.input_width, args.input_height,
    args.output_width, args.output_height,
    font_paths, selectable_colors, chars,
    args.max_select_count, entry_count=args.num_images)


def legacy():
  parser = ArgumentParser()
  parser.add_argument('--num-images', dest='num_images',
    default=10000, type=int, help='number of images to add to database')
  parser.add_argument('--input-width', dest='input_width',
    default=160, type=int, help='input image width')
  parser.add_argument('--input-height', dest='input_height',
    default=120, type=int, help='input image height')
  parser.add_argument('--output-width', dest='output_width',
    default=16, type=int, help='output image width')
  parser.add_argument('--output-height', dest='output_height',
    default=12, type=int, help='output image height')
  parser.add_argument('--characters', dest='characters',
    default='A,B,C,D,E,F,G,H,J,K', type=str,
    help='characters to create image from')

  parser.add_argument('-o', dest='output_db',
    default='yolo.sqlite3', type=str, help='output sqlite database path')
  parser.add_argument('--table-name', dest='table_name',
    default='yolo', type=str, help='output sqlite data table name')
  parser.add_argument('--max-count', dest='max_select_count',
    default=6, type=int, help='max number of characters in 1 image')
  args = parser.parse_args()

  chars = args.characters.split(',')
  selectable_colors = [
      (255, 255, 255), (255, 0, 0), (0, 255, 0),
      (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

  create_db(args.output_db,
      args.input_width,
      args.input_height,
      args.num_images,
      chars, selectable_colors)
  images, coordinates, dimensions, classes = \
    load_data(args.output_db, chars,
    args.output_width, args.output_height, 1)
  print('data shape: %s' % (str(images.shape)))
  print('coordinates shape: %s' % (str(coordinates.shape)))
  print('dimensions shape: %s' % (str(dimensions.shape)))
  print('classes shape: %s' % (str(classes.shape)))

  index = random.randrange(len(images))
  plt.figure()
  plt.imshow(images[index])
  plt.figure()
  plt.imshow(cv2.resize(classes[index],
      (int(args.image_width), int(args.image_height))))
  plt.show()


if __name__ == '__main__':
  main()

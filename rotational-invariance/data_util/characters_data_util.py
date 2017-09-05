import os
import sqlite3
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from argparse import ArgumentParser


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(folder):
  folder_path = os.path.expanduser(folder)
  images = []
  labels = []
  if os.path.isdir(folder_path):
    logger.info('load data from %s...' % (folder_path))
    images_folder = os.listdir(folder_path)
    for image_folder in images_folder:
      image_files = os.listdir(os.path.join(folder_path, image_folder))
      for image_file in image_files:
        image_path = os.path.join(folder_path, image_folder, image_file)
        img = np.array(Image.open(image_path))
        images.append(img)
        labels.append(image_folder)
  else:
    logger.info('%s is not found' % (folder_path))
  return images, labels


def load_background_image(path):
  if os.path.isfile(path):
    image = np.array(Image.open(path))
    return image
  else:
    logger.warning('%s not found' % (path))
  return None


def create_distroted_image(image, background_image,
    rotation, shear, padding, output_size):
  w = image.shape[1]
  h = image.shape[0]
  src = np.array([
    [0.0, 0.0],
    [w, 0.0],
    [0.0, h]], dtype=np.float32)
  dst = np.array([
    [0.0, 0.0],
    [w, 0.0],
    [shear, h]], dtype=np.float32)
  M = cv2.getAffineTransform(src, dst)
  with_alpha = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
  sheared = cv2.warpAffine(with_alpha, M,
    (w + shear, h))
  sheared = Image.fromarray(sheared)

  rotated = sheared.rotate(rotation, expand=True)
  output = Image.new('RGBA',
    (rotated.width + 2 * padding, rotated.height + 2 * padding),
    (0, 0, 0, 0))

  background = Image.fromarray(background_image)
  background_w = np.random.randint(1, background.width)
  background_h = np.random.randint(1, background.height)
  background_x = np.random.randint(0, background.width - background_w)
  background_y = np.random.randint(0, background.height - background_h)
  #  logger.info('(%d %d %d %d)' %
  #    (background_x, background_y,
  #     background_x + background_w,
  #     background_y + background_h))
  background_crop = background.crop((background_x, background_y,
    background_x + background_w, background_y + background_h))
  output.paste(background_crop.resize((output.width, output.height)))
  output.paste(rotated, (padding, padding), rotated)
  output = output.resize((output_size, output_size))
  return np.array(output)


def distorted_image(image, background_image, output_size, character_label,
    min_rotation=0, max_rotation=360, rotation_delta=20,
    min_shear=0, max_shear=60, shear_delta=15,
    min_padding=0, max_padding=60, padding_delta=15):
  distorted_images = []
  labels = []
  for r in range(min_rotation, max_rotation + rotation_delta, rotation_delta):
    for s in range(min_shear, max_shear + shear_delta, shear_delta):
      for p in range(min_padding, max_padding + padding_delta, padding_delta):
        di = create_distroted_image(image, background_image,
          -r, s, p, output_size)
        distorted_images.append(di)
        labels.append([character_label, r])
  return distorted_images, labels


def save_distorted(images, image_labels, background_image, output_size,
    output_db, table_name):
  connection = sqlite3.connect(output_db)
  cursor = connection.cursor()
  cursor.execute("""CREATE TABLE IF NOT EXISTS %s(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB NOT NULL,
    imageWidth INTEGER NOT NULL,
    imageHeight INTEGER NOT NULL,
    imageChannel INTEGER NOT NULL,
    char TEXT NOT NULL,
    rotation FLOAT NOT NULL);""" % (table_name))
  for i in range(len(images)):
    distorted_images, distorted_labels = distorted_image(images[i],
      background_image, output_size, image_labels[i])
    logger.info('output %s samples' % (len(distorted_images)))
    for j in range(len(distorted_images)):
      img = distorted_images[j]
      label = distorted_labels[j]
      cursor.execute("""INSERT INTO %s(image,
        imageWidth, imageHeight, imageChannel,
        char, rotation) VALUES(?, ?, ?, ?, ?, ?)""" % (table_name),
        (buffer(img), img.shape[1], img.shape[0], img.shape[2],
          label[0], label[1],))
      cv2.imshow('sample image', cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (300, 300)))
      cv2.waitKey(20)
    connection.commit()
  connection.close()


def main():
  parser = ArgumentParser()
  parser.add_argument('--folder', dest='folder',
    help='folders to load')
  parser.add_argument('--background-image', dest='background_image',
    help='background image to use')
  parser.add_argument('--output-image-size', dest='output_image_size',
    default=32, help='output image size')
  parser.add_argument('--output-db', dest='output_db',
    default='characters.sqlite3')
  parser.add_argument('--table-name', dest='table_name',
    default='characters', help='output table name')
  args = parser.parse_args()

  images, labels = load_data(args.folder)
  background_image = load_background_image(args.background_image)
  save_distorted(images, labels, background_image,
    args.output_image_size, args.output_db, args.table_name)


if __name__ == '__main__':
  main()

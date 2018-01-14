from __future__ import absolute_import

import sqlite3
import urllib2
import os
import logging
import zipfile
import tarfile
import struct
from StringIO import StringIO
from PIL import Image
from scipy.io import loadmat
from argparse import ArgumentParser


URL = 'http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/'
DATA_DIR = 'data'

logging.basicConfig()
logger = logging.getLogger('prepare')
logger.setLevel(logging.INFO)


def download_data(dataset):
  if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)

  data_path = os.path.join(DATA_DIR, dataset)
  if not os.path.isfile(data_path):
    logger.info('downloading %s...', dataset)

    url = URL + dataset
    content = urllib2.urlopen(url).read()
    logger.info('writing to %s...', data_path)
    with open(data_path, 'wb') as f:
      f.write(content)
  return data_path


def prepare_label(label_zip):
  label = {}

  annotations = os.path.join(DATA_DIR, 'annotations')
  if not os.path.isdir(annotations):
    with zipfile.ZipFile(label_zip) as f:
      f.extractall(DATA_DIR)

  for s in os.listdir(annotations):
    set_path = os.path.join(annotations, s)
    label[s] = {}

    for mat_file in os.listdir(set_path):
      label[s][mat_file] = {}

      mf_path = os.path.join(set_path, mat_file)
      vbb = loadmat(mf_path)
      num_frames = int(vbb['A'][0][0][0][0][0])
      obj_list = vbb['A'][0][0][1][0]
      #  maxObj   = int(vbb['A'][0][0][2][0][0])
      #  objInit  = vbb['A'][0][0][3][0]
      #  objLbl   = [str(v[0]) for v in vbb['A'][0][0][4][0]]
      #  objStr   = vbb['A'][0][0][5][0]
      #  objEnd   = vbb['A'][0][0][6][0]
      #  objHide  = vbb['A'][0][0][7][0]
      #  altered  = int(vbb['A'][0][0][8][0][0])
      #  log      = vbb['A'][0][0][9][0]
      #  logLen   = int(vbb['A'][0][0][10][0][0])
      logger.info('%s has %d frames', mat_file, num_frames)

      for frame_id, obj in enumerate(obj_list):
        objects_pos = []
        if len(obj) > 0:
          for id, pos, occl, lock, posv in zip(
              obj['id'][0], obj['pos'][0],
              obj['occl'][0], obj['lock'][0], obj['posv'][0]):
            #  keys = obj.dtype.names
            #  id   = int(id[0][0]) - 1  # MATLAB is 1-origin
            pos  = pos[0].tolist()
            #  occl = int(occl[0][0])
            #  lock = int(lock[0][0])
            #  posv = posv[0].tolist()
            objects_pos.append(pos)
        label[s][mat_file][frame_id] = objects_pos

  return label


def detect_format(image_format):
  if image_format == 100 or image_format == 200:
    return ".raw"
  elif image_format == 101:
    return ".brgb8"
  elif image_format == 102 or image_format == 201:
    return ".jpg"
  elif image_format == 103:
    return ".jbrgb"
  elif image_format == 1 or image_format == 2:
    return ".png"
  else:
    print "Invalid extension format " + image_format
    return None


def prepare_data(cursor, data_tar, label):
  folder = data_tar[:-4]
  if not os.path.isdir(folder):
    logger.info('extracting %s...', data_tar)
    with tarfile.open(data_tar) as f:
      f.extractall(DATA_DIR)


  for clip in os.listdir(folder):
    clip_path = os.path.join(folder, clip)
    logger.info('loading %s...', clip_path)

    with open(clip_path, 'rb') as f:
      SKIP = 28 + 8 + 512
      f.seek(SKIP)

      header_info = ["width", "height", "imageBitDepth",
        "imageBitDepthReal", "imageSizeBytes",
        "imageFormat", "numFrames"]
      header = {}

      for attr in header_info:
        header[attr] = struct.unpack('I', f.read(4))[0]

      # skip 4 bytes
      f.read(4)

      header["trueImageSize"] = struct.unpack('I', f.read(4))[0]
      header["fps"] = struct.unpack('d', f.read(8))[0]

      #  print header
      #  ext = detect_format(header["imageFormat"])

      # skip to image data
      f.seek(432, 1)
      for img_id in range(header["numFrames"]):
        img_size = struct.unpack('I', f.read(4))[0]

        img_data = f.read(img_size)
        #  img_name = str(img_id) + ext
        l = str(label[folder.split('/')[-1]][clip[:-4] + '.vbb'][img_id])
        cursor.execute("""INSERT INTO images VALUES(?, ?)""",
          (buffer(img_data), l,))

        # skip to next image
        f.seek(12, 1)


def save_data(dbname, dataset, label):
  connection = sqlite3.connect(dbname)
  cursor = connection.cursor()
  cursor.execute("""DROP TABLE IF EXISTS images;""")
  cursor.execute("""CREATE TABLE images(
    image BLOB NOT NULL,
    label TEXT NOT NULL);""")

  for ds in dataset:
    data = download_data(ds)
    prepare_data(cursor, data, label)

    connection.commit()
  connection.close()

def main():
  parser = ArgumentParser()
  parser.add_argument('--dbname', dest='dbname', default='pedestrian.sqlite3',
    type=str, help='output dbname')
  args = parser.parse_args()

  label_zip = download_data('annotations.zip')
  label = prepare_label(label_zip)

  datasets = [('set%02d.tar' % i) for i in range(11)]
  save_data(args.dbname, datasets, label)


if __name__ == '__main__':
  main()

import sqlite3
import os
import numpy as np
import time
import cv2
from argparse import ArgumentParser

from yolo_data_util import load_yolo_data


def main():
  parser = ArgumentParser()
  parser.add_argument('-i', dest='input_db', default='yolo.sqlite3',
    type=str, help='input sqlite3 dataset')
  parser.add_argument('--table-name', dest='table_name', default='yolo',
    type=str, help='data table name')
  args = parser.parse_args()

  images, labels = load_yolo_data(args.input_db, args.table_name)
  print(images.shape, labels.shape)
  for i in range(len(images)):
    label_display = labels[i, :, :, 0].reshape(labels.shape[1], labels.shape[2])
    indicator = label_display > 0
    label_display[indicator] = 255
    label_display.astype(np.uint8)
    #  print(labels[i, indicator, 1:])
    roi = images[i, :].copy()
    center = (labels[i, indicator, 1:3] *
      np.array([images.shape[2], images.shape[1]])).astype(np.int32)
    size = (labels[i, indicator, 3:4] * images.shape[2]).astype(np.int32)
    direction = (labels[i, indicator, 5]).reshape([-1, 1])
    direction -= np.pi / 2
    sin = np.sin(direction)
    cos = np.cos(direction)
    diff = size * np.concatenate([cos, sin], axis=1)
    p = (center + diff).astype(np.int32)
    for j in range(len(center)):
      cv2.circle(roi, (center[j][0], center[j][1]),
        size[j], (0, 255, 0), 2)
      cv2.line(roi, (center[j][0], center[j][1]),
        (p[j][0], p[j][1]), (0, 0, 255), 1)

    display_size = (640, 480)
    cv2.imshow('images',
      cv2.resize(cv2.cvtColor(images[i, :], cv2.COLOR_RGB2BGR), display_size))
    cv2.imshow('labels',
      cv2.resize(label_display, display_size))
    cv2.imshow('roi',
      cv2.resize(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR), display_size))

    key = cv2.waitKey(10000)
    if key in [ord('q'), ord('Q'), 27]:
      break


if __name__ == '__main__':
  main()

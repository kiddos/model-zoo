import cv2
import numpy as np


def threshold_characters(frame, params, min_area):
  width = frame.shape[1]
  height = frame.shape[0]
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  hsv_output = np.zeros(shape=[height, width, 1], dtype=np.uint8)
  roi = []
  filtered = np.copy(frame)
  roi_output = np.copy(frame)
  for color in params:
    threshold = np.zeros(shape=[height, width, 1], dtype=np.uint8)
    hlow = params[color]['HLow']
    hhigh = params[color]['HHigh']
    slow = params[color]['SLow']
    shigh = params[color]['SHigh']
    vlow = params[color]['VLow']
    vhigh = params[color]['VHigh']
    cv2.inRange(hsv, (hlow, slow, vlow),
      (hhigh, shigh, vhigh), threshold)
    cv2.imshow('threshold', threshold)
    #  morph = cv2.erode(cv2.dilate(threshold,
    #    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
    #    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    morph = cv2.dilate(cv2.erode(threshold,
      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))),
      cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))
    threshold = np.expand_dims(morph, axis=-1)
    _, contours, hiearchy = cv2.findContours(threshold,
      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
      area = cv2.contourArea(contour)
      if area >= min_area:
        bx, by, bw, bh = cv2.boundingRect(contour)
        source = [bx, by, bw, bh]
        roi.append(source)
        cv2.drawContours(filtered, contours, i, (0, 0, 255), 2)
        cv2.rectangle(roi_output, (int(bx), int(by)),
          (int(bx + bw), int(by + bh)), (0, 255, 0), 3)
        #  rect = cv2.minAreaRect(contour)
        #  box = cv2.boxPoints(rect)
        #  box = np.int0(box)
        #  cv2.drawContours(display2, [box], 0, (0, 255, 0), 2)
    hsv_output |= threshold
  filtered[(hsv_output == 0).squeeze(), :] = 0
  return hsv_output, filtered, roi_output, roi

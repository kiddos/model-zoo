import cv2
import numpy as np
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras


IMAGE_SIZE = 300
FONT = cv2.FONT_HERSHEY_SIMPLEX


def setup_gpu():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      print(e)


def load_model(path):
  print('loading model from ', path, '...')
  #  model = keras.models.load_model(path)
  #  model.summary()
  return tf.saved_model.load(path)


def display_class(c):
  if c == 0:
    return 'with mask', (113, 234, 124)
  elif c == 1:
    return 'mask weared incorrect', (234, 179, 113)
  else:
    return 'no mask', (106, 106, 236)



def main():
  setup_gpu()

  #  model = load_model('./version4.1')
  #  model = load_model('./version5.1')
  #  model = load_model('./version6.1')
  model = load_model('./version7.1')

  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print('camera not opened')
    return

  while True:
    ret, frame = cap.read()
    frame_copy = np.copy(frame)
    frame_size = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, [IMAGE_SIZE, IMAGE_SIZE])
    img_tensor = np.reshape(resized, [1, IMAGE_SIZE, IMAGE_SIZE, 3]).astype(np.float32)
    prediction = model(img_tensor, training=False)
    for row in prediction[0]:
      if row[0] >= 0.8:
        y1 = int(row[1] * frame_size[0])
        x1 = int(row[2] * frame_size[1])
        y2 = int(row[3] * frame_size[0])
        x2 = int(row[4] * frame_size[1])
        c = np.argmax(row[5:])
        text, color = display_class(c)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        display_text = '{}: {:2.2}'.format(text, row[0])
        cv2.putText(frame, display_text, (x1, y1), FONT, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('preview', frame)
    key = cv2.waitKey(6) & 0xFF
    if key == ord('q'):
      break
    elif key == ord('t'):
      now = datetime.now()
      t = now.strftime('%H:%M:%S')
      cv2.imwrite('capture-{}.png'.format(t), frame_copy)
      print('capture at ', t)

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()

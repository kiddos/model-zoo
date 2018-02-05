from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3


def analyse(dbname):
  connection = sqlite3.connect(dbname)
  cursor = connection.cursor()

  cursor.execute("""SELECT count(image) FROM images;""")
  print('number of images: ', cursor.fetchone()[0])

  cursor.execute("""SELECT min(width) FROM images;""")
  print('min width: ', cursor.fetchone()[0])
  cursor.execute("""SELECT max(width) FROM images;""")
  print('max width: ', cursor.fetchone()[0])

  cursor.execute("""SELECT min(height) FROM images;""")
  print('min height: ', cursor.fetchone()[0])
  cursor.execute("""SELECT max(height) FROM images;""")
  print('max height: ', cursor.fetchone()[0])

  cursor.execute("""SELECT label FROM images GROUP BY label;""")
  groups = cursor.fetchall()
  print('number of types of whales: ', len(groups))

  return len(groups)


analyse('./humback-whale.sqlite3')

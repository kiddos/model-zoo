import threading
import unittest
from argparse import ArgumentParser
from collections import deque
from PIL import Image

from plant_loader import PlantLoader


class PlantSampler(object):
  def __init__(self, num_workers, dbname, input_size, batch_size, load_all=False):
    self.running = False
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.load_all = load_all

    self._batch_data = deque(maxlen=100)
    self._loader = PlantLoader(dbname, input_size)
    self._loader.load_data()

    self.workers = []
    for _ in range(self.num_workers):
      self.workers.append(threading.Thread(target=self.sampling))

  def __del__(self):
    self.stop()

  def start(self):
    self.running = True
    for i in range(self.num_workers):
      self.workers[i].start()

  def stop(self):
    self.running = False
    try:
      for i in range(self.num_workers):
        self.workers[i].join()
    except: pass

  def queue_size(self):
    return len(self._batch_data)

  def sampling(self):
    while self.running:
      if len(self._batch_data) < 100:
        self._batch_data.append(
          self._loader.sample(self.batch_size, self.load_all))

  def get_data(self):
    while len(self._batch_data) == 0: pass
    return self._batch_data.popleft()

  def get_validation_data(self):
    return self._loader.get_validation_data(), \
      self._loader.get_validation_labels()


class TestPlantSampler(unittest.TestCase):
  def setUp(self):
    parser = ArgumentParser()
    parser.add_argument('--dbname', dest='dbname', default='plants.sqlite3',
      type=str, help='db to load')
    args = parser.parse_args()

    self.sampler = PlantSampler(128, args.dbname, 64, 32)

  def test_run(self):
    self.sampler.start()

    for i in range(100):
      data, label = self.sampler.get_data()
      self.assertEqual(data.shape, (32, 64, 64, 3))
      self.assertEqual(label.shape, (32, 12))
      #  print(self.sampler.queue_size())

    self.sampler.stop()


if __name__ == '__main__':
  unittest.main()

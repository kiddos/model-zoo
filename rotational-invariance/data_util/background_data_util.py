import urllib2
import os
import logging
import tarfile


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download():
  target = 'iccv09Data.tar.gz'
  url = 'http://dags.stanford.edu/data/iccv09Data.tar.gz'
  if not os.path.isfile(target):
    logger.info('downloading %s...' % (target))
    with open(target, 'wb') as f:
      data = urllib2.urlopen(url).read()
      f.write(data)
  else:
    logger.info('%s already exists.' % (target))
  return target


def extract(target):
  folder = 'iccv09Data'
  if not os.path.isdir(folder):
    logger.info('extracting %s' % (folder))
    with tarfile.open(target, 'r') as tf:
      tf.extractall()
  else:
    logger.info('%s exists.' % (folder))
  folder = os.path.join(folder, 'images')
  return [os.path.join(folder, f) for f in os.listdir(folder)]


def load():
  tar = download()
  images = extract(tar)
  return images

def main():
  images = load()
  print(images)


if __name__ == '__main__':
  main()

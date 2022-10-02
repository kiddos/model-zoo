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
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner) 
          
      
      safe_extract(tf)
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

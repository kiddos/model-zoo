from ftplib import FTP
import logging
import os
import tarfile


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download():
  tar = 'fonts.tar.gz'
  if not os.path.isfile(tar):
    logger.info('downloading %s' % (tar))
    ftp = FTP(host='140.127.205.173', user='icalpublic', passwd='nukical406')
    ftp.cwd('/sda1/public/data/rotation-fonts')
    with open(tar, 'wb') as f:
      ftp.retrbinary('RETR ' + tar, f.write)
  else:
    logger.info('%s already exists' % (tar))
  return tar


def extract(target):
  folder = 'fonts'
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
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(tf)
  else:
    logger.info('%s exists.' % (folder))
  folders = os.listdir(folder)
  fonts = []
  for f in folders:
    fonts += [os.path.join(folder, f, file)
        for file in os.listdir(os.path.join(folder, f)) if
        file.endswith('.ttf') and file.find('Italic') < 0]
  return fonts


def load():
  tar = download()
  fonts = extract(tar)
  return fonts


def main():
  fonts = load()
  print(fonts)


if __name__ == '__main__':
  main()

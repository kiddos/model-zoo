#!/usr/bin/env sh

if [ ! -d "duckymomo" ]; then
  echo "Downloading Data..."
  wget -r --ask-password --user=icalpublic ftp://140.127.205.173/sda1/public/data/duckymomo
  mv 140.127.205.173/sda1/public/data/duckymomo duckymomo
  rm -r 140.127.205.173
  clear
  echo "Ducky momo is my best friend!!"
else
  echo "Data already downloaded."
fi

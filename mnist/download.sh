#!/usr/bin/env sh

if [ ! -f "train.csv" ]; then
  wget --user="icalpublic" --ask-password "ftp://140.127.205.173/sda1/public/data/mnist/train.csv"
fi

if [ ! -f "test.csv" ]; then
  wget --user="icalpublic" --ask-password "ftp://140.127.205.173/sda1/public/data/mnist/test.csv"
fi

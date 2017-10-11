#!/usr/bin/env bash

if [ ! -f cohn-kanade.tgz ]; then
  echo "downloading cohn-kanade.tgz ..."
  wget --user $1 --password $2 http://www.consortium.ri.cmu.edu/data/ck/CK/cohn-kanade.tgz
else
  echo "cohn-kanade.tgz already downloaded"
fi

files=("Landmarks.zip" "Emotion_labels.zip"	"FACS_labels.zip"
"ReadMeCohnKanadeDatabase_website.txt" "extended-cohn-kanade-images.zip")
for f in ${files[*]}; do
  if [ ! -f $f ]; then
    echo "Downloading $f ..."
    wget --user $1 --password $2 http://www.consortium.ri.cmu.edu/data/ck/CK+/$f
  else
    echo "$f downloaded."
  fi
done

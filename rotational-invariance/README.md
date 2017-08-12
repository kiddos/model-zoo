Rotational Invariance
=====================

This is an experiment on rotational invariance.
We use the [Stanford background data](http://dags.stanford.edu/projects/scenedataset.html) and some fonts found at [google font](https://fonts.google.com/) to create random input images.

We start with simple model that inputs an image with only character(eg. A, B, C, D, E, F, G, H, J, K) and predict which character is it and rotation.

We also build a tiny yolo model that predict the roi of characters in an image

The next step, we try to combine 2 models together, which inputs an image and
predict the roi of character, its classes and its rotation. Unfortunately, our GTX980 cannot train a large enough model that simulateously predicts all of these attributes.

Currently, we are working on simpler version of it, which we limit the image
background to the area where our robotic arm can reach.


## Rotational Invariance Model

TODO

## Tiny YOLO for locating characters

TODO

## YOLO for locating Characters and classify them and predict its rotation

TODO

## Setup

install OpenCV with python

run the command to install other dependencies and create protos

```shell
./setup.sh
```

## Prepare Data

Stanford background data

```shell
cd data_util
python background_data_util.py
```

Data we collected

```shell
cd data_util
download_ftp_images.sh
```

## Inference Server

generate protos

```shell
./setup.sh
```

launch

```shell
python roi_server.py
```

## Inference Client Test

TODO

Plant Recognizer
================

## Download

```
kg download -u <username> -p <password> -c plant-seedlings-classification -f sample_submission.zip

kg download -u <username> -p <password> -c plant-seedlings-classification -f train.zip

kg download -u <username> -p <password> -c plant-seedlings-classification -f test.zip
```

## Prepare Data

```
python plant_prepare.py --width 32 --height 32 --dbname plants-32x32.sqlite3
```

## Train

```
python plant_recognizer.py --save-epoch 200 --max-epoches 200000 --learning-rate
1e-3 --keep-prob 0.8 --display-epoch 100 --batch-size 256 --dbname
plants-32x32.sqlite3
```

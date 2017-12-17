Titanic Disaster Prevention
===========================

## Download data

```
kg download -u kiddos -p a3366630 -c titanic -f train.csv
kg download -u kiddos -p a3366630 -c titanic -f test.csv
```

## Train the model

```
python titanic_fnn.py --saving True
```

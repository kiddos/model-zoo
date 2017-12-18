Titanic Disaster Prevention
===========================

## Download data

```
kg download -u <username> -p <password> -c titanic -f train.csv
kg download -u <username> -p <password> -c titanic -f test.csv
```

## Train the model

```
python titanic_fnn.py --saving True
```

import pandas as pd
import sqlite3


connection = sqlite3.connect('league.sqlite3')
cursor = connection.cursor()

dataframe = pd.read_csv('./data/matchinfo.csv')
for c in dataframe:
  print c

champs_col = [
  'blueTopChamp',
  'blueJungleChamp',
  'blueMiddleChamp',
  'blueADCChamp',
  'blueSupportChamp',
  'redTopChamp',
  'redJungleChamp',
  'redMiddleChamp',
  'redADCChamp',
  'redSupportChamp']

champs = []
for c in dataframe[champs_col].as_matrix().reshape([-1]):
  if c not in champs:
    champs.append(c)
champs = {c: i for i, c in enumerate(sorted(champs))}
print(champs)
print(len(champs))

cursor.execute("""DROP TABLE IF EXISTS champs;""")
cursor.execute("""CREATE TABLE champs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(256) NOT NULL);""")
for c in champs:
  cursor.execute("""INSERT INTO champs(name) VALUES(?)""", (c,))


result = dataframe[['bResult', 'rResult']].as_matrix()
assert (result[result[:, 0] == 1, 1] == 0).all()
assert (result[result[:, 1] == 1, 0] == 0).all()


data_col = champs_col + ['bResult', 'rResult']


def insert(data, cursor, tablename):
  cursor.execute("""DROP TABLE IF EXISTS %s;""" % (tablename))
  cursor.execute("""CREATE TABLE %s(
    blueTopChamp INTEGER NOT NULL,
    blueJungleChamp INTEGER NOT NULL,
    blueMiddleChamp INTEGER NOT NULL,
    blueADCChamp INTEGER NOT NULL,
    blueSupportChamp INTEGER NOT NULL,
    redTopChamp INTEGER NOT NULL,
    redJungleChamp INTEGER NOT NULL,
    redMiddleChamp INTEGER NOT NULL,
    redADCChamp INTEGER NOT NULL,
    redSupportChamp INTEGER NOT NULL,
    bResult INTEGER NOT NULL,
    rResult INTEGER NOT NULL)""" % (tablename))

  for raw_entry in data:
    entry = [champs[r] if r in champs else r for r in raw_entry]
    cursor.execute("""INSERT INTO %s
      VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""" % (tablename), entry)


raw_data = dataframe[data_col].sample(frac=1).as_matrix()
valid_index = int(len(raw_data) * 0.8)
train_data = raw_data[:valid_index]
valid_data = raw_data[valid_index:]

insert(train_data, cursor, 'matches_train')
insert(valid_data, cursor, 'matches_valid')

connection.commit()
connection.close()

import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('./raw_data/reviews_Cell_Phones_and_Accessories_5.json.gz')

df.to_csv('reviews_Cell_Phones_and_Accessories_5.csv', index=False)
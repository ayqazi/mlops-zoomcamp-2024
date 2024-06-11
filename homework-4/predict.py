#!/usr/bin/env python
# coding: utf-8

import pickle
import sys

import pandas as pd

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

if len(sys.argv) < 3:
    print('Usage: predict.py year month', file=sys.stderr)
    exit(1)

year, month = int(sys.argv[1]), int(sys.argv[2])

if year < 2009 or year > 2023:
    print('Year must be between 2009 and 2023', file=sys.stderr)
    exit(2)

if month < 1 or month > 12:
    print('Month must be between 1 and 12', file=sys.stderr)
    exit(3)

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
# print(df.head())

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
print(f'Mean prediction: {y_pred.mean()}')

# df['ride_id'] = f'2023/03_' + df.index.astype('str')
# df_result = pd.concat([df['ride_id'], pd.DataFrame(y_pred, columns=["predictions"], index=df.index)], axis=1)
# df_result.to_parquet(
#     "../tmp/df_result.parquet",
#     engine='pyarrow',
#     compression=None,
#     index=False
# )

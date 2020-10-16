from pathlib import Path

from x04_train_test_split import DEV_SET_REVIEWS_JOB, TRAIN_SET_REVIEWS_JOB

# import pandas as pd


DATA_DIR = Path('data')
keep_cols = ['overall', 'reviewerID', 'item', 'reviewDate']

train = TRAIN_SET_REVIEWS_JOB.load_parquet()
train_tuples = train.compute()
train_tuples = train_tuples[keep_cols].reset_index(drop=True)
train_tuples.to_feather(DATA_DIR / "train_tuples.f")

test = DEV_SET_REVIEWS_JOB.load_parquet()
test_tuples = test.compute()
test_tuples = test_tuples[keep_cols].reset_index(drop=True)
test_tuples.to_feather(DATA_DIR / "test_tuples.f")

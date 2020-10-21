import os
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("data/raw/amazon")
PROCESSED_DATA_DIR = Path("data/processed/amazon")
YEARS_BACK = 5
MIN_REVIEWS_PER_USER = 5

reviews = pd.read_feather(RAW_DATA_DIR / "Movies_and_TV.f")

reviews["reviewDate"] = pd.to_datetime(reviews["unixReviewTime"], unit="s")

start_date = reviews.reviewDate.max() - pd.DateOffset(years=YEARS_BACK)
recent_reviews = reviews[reviews.reviewDate >= start_date]

user_counts = reviews.reviewerID.value_counts()
user_counts = user_counts[user_counts >= MIN_REVIEWS_PER_USER].reset_index()
user_counts.columns = ["reviewerID", "counts"]

recent_reviews = recent_reviews[recent_reviews.reviewerID.isin(user_counts.reviewerID)]

train_size = recent_reviews.shape[0] - round(recent_reviews.shape[0] * 0.2)
train_reviews = recent_reviews.iloc[:train_size]
test_reviews = recent_reviews.iloc[train_size:]

valid_size = round(test_reviews.shape[0] * 0.5)
valid_reviews = test_reviews.iloc[:valid_size]
test_reviews = test_reviews.iloc[:valid_size]

train_reviews.reset_index(drop=True, inplace=True)
valid_reviews.reset_index(drop=True, inplace=True)
test_reviews.reset_index(drop=True, inplace=True)

if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

train_reviews.to_feather(PROCESSED_DATA_DIR / "train.f")
valid_reviews.to_feather(PROCESSED_DATA_DIR / "valid.f")
test_reviews.to_feather(PROCESSED_DATA_DIR / "test.f")

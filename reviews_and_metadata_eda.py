# import gc
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("raw_data")
MIN_REVIEWS_PER_USER = 5
N_YEARS_BACK = 5

movies_and_tv = pd.read_feather(RAW_DATA_DIR / "Movies_and_TV.f")

user_counts = movies_and_tv.reviewerID.value_counts()
user_counts = user_counts[user_counts >= MIN_REVIEWS_PER_USER].reset_index()
user_counts.columns = ["reviewerID", "counts"]

movies_and_tv_sample = movies_and_tv[
    movies_and_tv.reviewerID.isin(user_counts.reviewerID)
]

movies_and_tv_sample["reviewDate"] = pd.to_datetime(
    movies_and_tv_sample["unixReviewTime"], unit="s"
)

start_date = movies_and_tv_sample.reviewDate.max() - pd.DateOffset(years=N_YEARS_BACK)
movies_and_tv_sample = movies_and_tv_sample[
    movies_and_tv_sample.reviewDate >= start_date
]

unique_users = movies_and_tv_sample.reviewerID.nunique()
unique_items = movies_and_tv_sample.asin.nunique()

sparsity = movies_and_tv_sample.shape[0] / (unique_users * unique_items)

releaseDate = movies_and_tv_sample.groupby('asin')['reviewDate'].min().reset_index()
releaseDate.columns = ['asin', 'releaseDate']

movies_and_tv_sample = movies_and_tv_sample.merge(releaseDate, on='asin')
movies_and_tv_sample.drop('image', axis=1, inplace=True)

meta_movies_and_tv = pd.read_pickle(RAW_DATA_DIR / "meta_Movies_and_TV.p")

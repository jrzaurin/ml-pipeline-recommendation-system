from pathlib import Path

import pandas as pd

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
RAW_DATA_DIR = Path("data/raw/amazon")
PROCESSED_DATA_DIR = Path("data/processed/amazon")

# Remember that here we have removed duplicates, which might be a customer
# buying the same item. However, the truth is that when you have a look to the
# duplicates they are mostly mistakes in the data collection. For example, out
# of all the users used in training there are only 1511 that buy more than one
# item in more than one diff date. I had a look to a few, and there are still
# mistakes in data collection. In summary, we will use directly the training
# dataset.
train = pd.read_feather(PROCESSED_DATA_DIR / "train_full.f")

train.drop(
    [
        c
        for c in train.columns
        if c
        not in ["overall", "verified", "user", "item", "reviewDate", "recency_factor"]
    ],
    axis=1,
    inplace=True,
)

# time diff between sessions, time of the day, day of the week, average score,
# most recent category, brand, language and detail, quantile price when
# possible and top N topics and/or dimensions
item_feat = pd.read_pickle(PROCESSED_DATA_DIR / "meta_movies_and_tv_processed.p")

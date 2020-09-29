import gc
# import json
from pathlib import Path

# import dask.bag as db
import pandas as pd


def flatten(record):
    return {k: v for k, v in record.items()}


RAW_DATA_DIR = Path("raw_data")

df = pd.read_json(RAW_DATA_DIR / "Movies_and_TV.json.gz", lines=True)
# feather, parquet...we will decide in due time
df.to_feather(RAW_DATA_DIR / "Movies_and_TV.f")
del df
gc.collect()

df = pd.read_json(RAW_DATA_DIR / "meta_Movies_and_TV.json.gz", lines=True)
df.to_pickle(RAW_DATA_DIR / "meta_Movies_and_TV.p")
del df
gc.collect()

# # or alternatively
# reviews = db.read_text(RAW_DATA_DIR / "Movies_and_TV.json.gz").map(json.loads)
# df = reviews.map(flatten).to_dataframe()
# df.to_parquet("raw_data/Movies_and_TV.parquet")
# del (reviews, df)
# gc.collect()

# reviews = db.read_text(RAW_DATA_DIR / "meta_Movies_and_TV.json.gz").map(json.loads)
# df = reviews.map(flatten).to_dataframe()
# df.to_parquet("raw_data/meta_Movies_and_TV.parquet")
# del (reviews, df)
# gc.collect()

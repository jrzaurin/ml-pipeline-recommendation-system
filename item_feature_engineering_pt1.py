import string
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
RAW_DATA_DIR = Path("data/raw/amazon")
PROCESSED_DATA_DIR = Path("data/processed/amazon")


def _extract_category(category):
    df_list = []
    for i, cat in enumerate(tqdm(category)):
        cats = pd.DataFrame(cat).transpose()
        df_list.append(cats)
    return pd.concat(df_list).reset_index(drop=True)


def process_category(df_inp):

    df = df_inp.copy()
    df["category"] = df.category.apply(lambda x: np.nan if x == [] else x)
    df_cat = df[~df.category.isna()].reset_index(drop=True)

    cats = _extract_category(df_cat.category)
    cats.columns = ["cat" + str(i) for i in range(cats.shape[1])]
    cats["item"] = df_cat.item

    # we use category 2
    cats = cats[["item", "cat2"]]
    cats.rename(columns={"cat2": "category"}, inplace=True)

    df.drop("category", axis=1, inplace=True)
    df = pd.merge(df, cats, on="item")
    df["category"] = df.category.str.lower()

    return df


def process_brand(df_inp):

    df = df_inp.copy()
    df["brand"] = df.brand.str.lower()
    df["brand"] = df.brand.apply(
        lambda x: np.nan
        if (
            x in string.punctuation
            or x in ["n/a", "na", "\n                "]
            or x == ""
            or x == " "
        )
        else x
    )
    return df


def process_price(df_inp):

    df = df_inp.copy()
    df["price"] = df.price.apply(
        lambda x: np.nan if (x == "" or x == " " or len(str(x)) > 15) else x
    )
    df["price"] = df.price.apply(
        lambda x: str(x).replace("$", "").replace(",", "")
    ).astype("float")
    return df


def _extract_language(detail):
    try:
        return detail["Language:"].split()[0].lower().replace(",", "")
    except Exception:
        return np.nan


def process_details(df_inp):

    df = df_inp.copy()
    df["language"] = df.details.apply(lambda x: _extract_language(x))
    df.drop("details", axis=1, inplace=True)
    return df


if __name__ == "__main__":

    active_items = pd.concat(
        [
            pd.read_feather(PROCESSED_DATA_DIR / "train_full.f"),
            pd.read_feather(PROCESSED_DATA_DIR / "valid_full.f"),
            pd.read_feather(PROCESSED_DATA_DIR / "test_full.f"),
        ]
    ).item.unique()
    meta = pd.read_pickle(RAW_DATA_DIR / "meta_Movies_and_TV.p")
    meta.rename(columns={"asin": "item"}, inplace=True)
    meta.drop_duplicates("item", keep="last", inplace=True)
    meta = meta[meta.item.isin(active_items)].reset_index(drop=True)

    # Drop columns with mostly NaN
    meta.drop(
        [
            "fit",
            "tech1",
            "tech2",
            "feature",
            "rank",
            "main_cat",
            "similar_item",
            "date",
            "image",
        ],
        axis=1,
        inplace=True,
    )

    meta_processed = process_category(meta)
    meta_processed = process_brand(meta_processed)
    meta_processed = process_price(meta_processed)
    meta_processed = process_details(meta_processed)

    meta_processed.to_pickle(PROCESSED_DATA_DIR / "item_features.p")

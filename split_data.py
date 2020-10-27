import gc
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

pd.options.display.max_columns = 100
RAW_DATA_DIR = Path("data/raw/amazon")
PROCESSED_DATA_DIR = Path("data/processed/amazon")


def sigmoid(x, xmid, tau, top):
    return top / (1.0 + np.exp(-(x - xmid) / tau))


def _compute_scores(df):
    df["score_binary"] = df.overall.apply(lambda x: 1 if x in [1, 2, 3] else 2)
    df["score_multic"] = df.overall.apply(
        lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3
    )


def _compute_recency_factor(df, xmid=365, tau=80, top=1):
    present = df.reviewDate.max()
    df["days_to_present"] = present - df["reviewDate"]
    df["days_to_present_inv"] = (df.days_to_present.max() - df.days_to_present).dt.days
    df["recency_factor"] = np.round(
        sigmoid(df.days_to_present_inv.values, xmid, tau, top), 2
    )
    df.drop(["days_to_present", "days_to_present_inv"], axis=1, inplace=True)


def _drop_low_interactions_users(df, min_reviews_per_user):
    user_counts = df.reviewerID.value_counts()
    user_counts = user_counts[user_counts >= min_reviews_per_user].reset_index()
    user_counts.columns = ["reviewerID", "counts"]
    df = df[df.reviewerID.isin(user_counts.reviewerID)]
    return df


def _keep_recent_n_years(df, n_years=5):
    df["reviewDate"] = pd.to_datetime(df["unixReviewTime"], unit="s")
    start_date = df.reviewDate.max() - pd.DateOffset(years=n_years)
    df_recent = df[df.reviewDate >= start_date]
    df_recent.drop_duplicates(["reviewerID", "asin"], inplace=True, keep="last")
    return df_recent


def time_train_test_split(df, dataset, test_fraction=0.2, n_years=5):
    df_recent = _keep_recent_n_years(df, n_years)
    df_recent.sort_values("reviewDate", inplace=True)

    train_size = df_recent.shape[0] - round(df_recent.shape[0] * test_fraction)
    train_df = df_recent.iloc[:train_size]
    test_df = df_recent.iloc[train_size:]
    valid_size = round(test_df.shape[0] * 0.5)
    valid_df = test_df.iloc[:valid_size]
    test_df = test_df.iloc[valid_size:]

    # scores and recency_factor are only needed for the training dataset,
    # since metrics on validation and testing will be ranking metrics
    train_df.reset_index(drop=True, inplace=True)
    _compute_scores(train_df)
    _compute_recency_factor(train_df)

    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    if not os.path.isdir(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    train_df.to_feather(PROCESSED_DATA_DIR / "_".join(["train", dataset + ".f"]))
    valid_df.to_feather(PROCESSED_DATA_DIR / "_".join(["valid", dataset + ".f"]))
    test_df.to_feather(PROCESSED_DATA_DIR / "_".join(["test", dataset + ".f"]))


def leave_one_out_train_test_split(df, n_years=5, min_reviews_per_user=7, n_neg=99):
    # based on Xiangnan He, et al, 2017 train/test split with implicit negative feedback

    # 1. sorting by date
    df_recent = _keep_recent_n_years(df, n_years)
    df_recent = _drop_low_interactions_users(df_recent, min_reviews_per_user)

    df_recent = df_recent.sort_values(
        ["reviewerID", "reviewDate"], ascending=[True, True]
    ).reset_index(drop=True)[["reviewerID", "asin", "overall"]]

    n_users = df_recent.reviewerID.nunique()
    n_items = df_recent.asin.nunique()

    # 2. use last ratings for validation/testing and all the previous for training
    test = df_recent.groupby("reviewerID").tail(2)
    train = pd.merge(
        df_recent, test, on=["reviewerID", "asin"], how="outer", datasetes=("", "_y")
    )
    train = train[train.overall_y.isnull()]
    train.drop("overall_y", axis=1, inplace=True)
    valid = test.groupby("reviewerID").head(1)
    test = test.groupby("reviewerID").tail(1)

    # select 99 random movies per user that were never rated by that user
    all_items = df_recent.asin.unique()

    def _sample_negative_test(group):
        rated_items = group.asin.tolist()
        non_rated_items = np.random.choice(np.setdiff1d(all_items, rated_items), n_neg)
        return np.hstack([group.reviewerID.unique(), non_rated_items])

    print("sampling not rated items...")
    start = time()
    with Pool(cpu_count()) as p:
        non_rated_items = p.map(
            _sample_negative_test,
            [gr for _, gr in df_recent.groupby("reviewerID")],
        )
    end = time() - start
    print("sampling took {} min".format(round(end / 60, 2)))

    cols = ["reviewerID"] + ["item_n" + str(i + 1) for i in range(99)]
    negative = pd.DataFrame(non_rated_items, columns=cols)
    negative = negative.melt("reviewerID")[["reviewerID", "value"]]
    negative.columns = ["reviewerID", "asin"]
    negative["overall"] = 0

    valid_negative = (
        pd.concat([valid[["reviewerID", "asin", "overall"]], negative])
        .sort_values(["reviewerID", "overall"], ascending=[True, False])
        .reset_index(drop=True)
    )

    test_negative = (
        pd.concat([test[["reviewerID", "asin", "overall"]], negative])
        .sort_values(["reviewerID", "overall"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Ensuring that the 1st element every 100 is the rated item. This is
    # fundamental for testing
    assert np.all(test_negative.overall.values[0::100] != 0) and np.all(
        valid_negative.overall.values[0::100] != 0
    )

    # Save
    np.savez(
        PROCESSED_DATA_DIR / "leave_one_out_train_test_split.npz",
        train=train,
        valid=valid_negative,
        test=test_negative,
        n_users=n_users,
        n_items=n_items,
    )


if __name__ == '__main__':

    print("INFO: splitting full dataset...")
    df = pd.read_feather(RAW_DATA_DIR / "Movies_and_TV.f")
    time_train_test_split(df, dataset="full")
    del df
    gc.collect()

    print("INFO: splitting 5 core dataset...")
    df_5core = pd.read_feather(RAW_DATA_DIR / "Movies_and_TV_5.f")
    time_train_test_split(df_5core, dataset="5core")
    del df_5core
    gc.collect()

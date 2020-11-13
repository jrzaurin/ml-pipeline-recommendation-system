import gc
import os
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

pd.options.display.max_columns = 100
RAW_DATA_DIR = Path("data/raw/amazon")
PROCESSED_DATA_DIR = Path("data/processed/amazon")


def rename_user_item_columns(df):
    return df.rename(columns={"reviewerID": "user", "asin": "item"})


def sigmoid(x, xmid, tau, top):
    return top / (1.0 + np.exp(-(x - xmid) / tau))


def compute_scores(df):
    df["score_binary"] = df.overall.apply(lambda x: 1 if x in [1, 2, 3] else 2)
    df["score_multic"] = df.overall.apply(
        lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3
    )


def compute_recency_factor(df, xmid=365, tau=80, top=1):
    present = df.reviewDate.max()
    df["days_to_present"] = present - df["reviewDate"]
    df["days_to_present_inv"] = (df.days_to_present.max() - df.days_to_present).dt.days
    df["recency_factor"] = np.round(
        sigmoid(df.days_to_present_inv.values, xmid, tau, top), 2
    )
    df.drop(["days_to_present", "days_to_present_inv"], axis=1, inplace=True)


def keep_last_n_years(df, n_years=5):
    df["reviewDate"] = pd.to_datetime(df["unixReviewTime"], unit="s")
    start_date = df.reviewDate.max() - pd.DateOffset(years=n_years)
    df_recent = df[df.reviewDate >= start_date]
    times_bought = df_recent.groupby(["user", "item"]).size().reset_index()
    times_bought.columns = ["user", "item", "times_bought"]
    df_recent.drop_duplicates(["user", "item"], inplace=True, keep="last")
    df_recent = pd.merge(df_recent, times_bought, on=["user", "item"])
    return df_recent


def filter_users_and_items(df_inp, min_reviews_per_user=5, min_reviews_per_item=1):
    df = df_inp.copy()
    if min_reviews_per_user > 1:
        keep_users = df.user.value_counts()[
            df.user.value_counts() >= min_reviews_per_user
        ].index
        df = df[df.user.isin(keep_users)]
    if min_reviews_per_item > 1:
        keep_items = df.item.value_counts()[
            df.item.value_counts() >= min_reviews_per_item
        ].index
        df = df[df.user.isin(keep_items)]
    return df.reset_index(drop=True)


def time_train_test_split(df_inp, dataset_name, test_fraction=0.2):
    """
    'traditional' train and test split based on a given fraction of the data
    """

    df = df_inp.copy()
    df.sort_values("reviewDate", inplace=True)

    train_size = df.shape[0] - round(df.shape[0] * test_fraction)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    valid_size = round(test_df.shape[0] * 0.5)
    valid_df = test_df.iloc[:valid_size]
    test_df = test_df.iloc[valid_size:]

    train_df.reset_index(drop=True, inplace=True)
    # recency_factor is only needed for the training dataset, since metrics on
    # validation and testing will be ranking metrics
    compute_recency_factor(train_df)
    compute_scores(train_df)

    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    compute_scores(valid_df)
    compute_scores(test_df)

    if not os.path.isdir(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    train_df.to_feather(PROCESSED_DATA_DIR / "_".join(["train", dataset_name + ".f"]))
    valid_df.to_feather(PROCESSED_DATA_DIR / "_".join(["valid", dataset_name + ".f"]))
    test_df.to_feather(PROCESSED_DATA_DIR / "_".join(["test", dataset_name + ".f"]))


def user_item_to_index(
    df_inp, user_idx_fname=None, item_idx_fname=None, users_idx=None, items_idx=None
):

    df = df_inp.copy()

    if users_idx is not None and items_idx is not None:
        df["user"] = df["user"].map(users_idx).astype(int)
        df["item"] = df["item"].map(items_idx).astype(int)
        return df

    assert (
        user_idx_fname is not None and item_idx_fname is not None
    ), "you forgot output file names"
    users = df.user.unique()
    items = df.item.unique()
    users_idx = {k: v for v, k in enumerate(users)}
    items_idx = {k: v for v, k in enumerate(items)}

    df["user"] = df["user"].map(users_idx).astype(int)
    df["item"] = df["item"].map(items_idx).astype(int)

    pickle.dump(users_idx, open(PROCESSED_DATA_DIR / user_idx_fname, "wb"))
    pickle.dump(items_idx, open(PROCESSED_DATA_DIR / item_idx_fname, "wb"))

    return df, users_idx, items_idx


def leave_one_out_train_test_split(df_inp, dataset_name):
    """
    split using all but one for training
    """

    df = df_inp.copy()
    df = df.sort_values(["user", "reviewDate"], ascending=[True, True]).reset_index(
        drop=True
    )[["user", "item", "reviewDate", "overall", "times_bought"]]

    test = df.groupby("user").tail(2)

    train = pd.merge(df, test, on=["user", "item"], how="outer", suffixes=("", "_y"))
    train = train[train.overall_y.isnull()].reset_index(drop=True)
    train.drop([c for c in train.columns if "_y" in c], axis=1, inplace=True)
    compute_recency_factor(train, xmid=730, tau=120, top=1)
    compute_scores(train)

    valid = test.groupby("user").head(1).reset_index(drop=True)
    test = test.groupby("user").tail(1).reset_index(drop=True)
    # this has to be done better, but for now, we live with it...
    compute_recency_factor(valid, xmid=730, tau=120, top=1)
    compute_scores(valid)
    compute_scores(test)

    if not os.path.isdir(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    train.to_feather(
        PROCESSED_DATA_DIR / "_".join(["leave_one_out_tr", dataset_name + ".f"])
    )
    valid.to_feather(
        PROCESSED_DATA_DIR / "_".join(["leave_one_out_val", dataset_name + ".f"])
    )
    test.to_feather(
        PROCESSED_DATA_DIR / "_".join(["leave_one_out_te", dataset_name + ".f"])
    )

    cols_to_return = ["user", "item", "overall"]
    return train[cols_to_return], valid[cols_to_return], test[cols_to_return]


def get_last_n(g, fr=0.2):
    return g.tail(int(g.shape[0] * fr))


def leave_n_out_train_test_split(df_inp, dataset_name, test_size=0.2):
    """
    split using all but the last X'%' for training
    """

    df = df_inp.copy()
    df = df.sort_values(["user", "reviewDate"], ascending=[True, True]).reset_index(
        drop=True
    )[["user", "item", "reviewDate", "overall", "times_bought"]]

    print("INFO: first round, train/test split")
    tr_user_groups = df.groupby("user")
    with Pool(cpu_count()) as p:
        res = p.map(partial(get_last_n, fr=test_size), [g for _, g in tr_user_groups])
    test_and_val = pd.concat(res).reset_index(drop=True)
    del res
    gc.collect()
    train = pd.merge(
        df, test_and_val, on=["user", "item"], how="outer", suffixes=("", "_y")
    )
    train = train[train.overall_y.isnull()].reset_index(drop=True)
    train.drop([c for c in train.columns if "_y" in c], axis=1, inplace=True)
    compute_recency_factor(train, xmid=730, tau=120, top=1)
    compute_scores(train)

    print("INFO: second round, valid/test split")
    te_user_groups = test_and_val.groupby("user")
    with Pool(cpu_count()) as p:
        te_res = p.map(partial(get_last_n, fr=0.5), [g for _, g in te_user_groups])
    test = pd.concat(te_res).reset_index(drop=True)
    valid = pd.merge(
        test_and_val, test, on=["user", "item"], how="outer", suffixes=("", "_y")
    )
    valid = valid[valid.overall_y.isnull()].reset_index(drop=True)
    valid.drop([c for c in valid.columns if "_y" in c], axis=1, inplace=True)
    compute_scores(valid)
    compute_scores(test)

    train.to_feather(
        PROCESSED_DATA_DIR / "_".join(["leave_n_out_tr", dataset_name + ".f"])
    )
    valid.to_feather(
        PROCESSED_DATA_DIR / "_".join(["leave_n_out_val", dataset_name + ".f"])
    )
    test.to_feather(
        PROCESSED_DATA_DIR / "_".join(["leave_n_out_te", dataset_name + ".f"])
    )

    cols_to_return = ["user", "item", "overall"]
    return train[cols_to_return], valid[cols_to_return], test[cols_to_return]


def sample_negative(group, all_items, n_neg):
    rated_items = group.item.tolist()
    non_rated_items = np.random.choice(np.setdiff1d(all_items, rated_items), n_neg)
    return np.hstack([group.user.unique(), non_rated_items])


def sample_negative_test(train, test, n_neg, strategy, dataset_name, is_valid):
    """
    Adding impicit negative feedback based on Xiangnan He, et al, 2017
    """
    user_idx_fname = "_".join(["user_idx", strategy, "w_negative", dataset_name])
    item_idx_fname = "_".join(["item_idx", strategy, "w_negative", dataset_name])
    dataset_fname = "_".join([strategy, "w_negative", dataset_name])

    if is_valid:
        user_idx_fname += "_valid.p"
        item_idx_fname += "_valid.p"
        dataset_fname += "_valid.npz"
    else:
        user_idx_fname += "_test.p"
        item_idx_fname += "_test.p"
        dataset_fname += "_test.npz"

    # numericalize has to happen here so sample_negative runs way faster
    test = test[test.item.isin(train.item.unique())]
    train, users_idx, items_idx = user_item_to_index(
        train, user_idx_fname, item_idx_fname
    )
    test = user_item_to_index(test, users_idx=users_idx, items_idx=items_idx)
    train_groups = train[train.user.isin(test.user.unique())].groupby("user")

    n_users = train.user.nunique()
    n_items = train.item.nunique()
    all_items = train.item.unique()

    print("INFO: sampling not rated items...")
    start = time()
    with Pool(cpu_count()) as p:
        non_rated_items = p.map(
            partial(sample_negative, all_items=all_items, n_neg=n_neg),
            [gr for _, gr in train_groups],
        )
    end = time() - start
    print("INFO: sampling took {} min".format(round(end / 60, 2)))

    negative = pd.DataFrame(non_rated_items)
    cols = ["user"] + ["col_n" + str(i + 1) for i in range(negative.shape[1] - 1)]
    negative.columns = cols
    negative = negative.melt("user")[["user", "value"]].rename(
        columns={"value": "item"}
    )
    negative.dropna(inplace=True)
    negative["overall"] = 0

    # Ensuring that the 1st element every 100 is the rated item. This is
    # fundamental for testing
    test_negative = (
        pd.concat([test[["user", "item", "overall"]], negative])
        .sort_values(["user", "overall"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Save
    np.savez(
        PROCESSED_DATA_DIR / dataset_fname,
        train=train,
        test=test_negative,
        n_users=n_users,
        n_items=n_items,
    )


if __name__ == "__main__":

    for dataset in ["full", "5core"]:
        if dataset == "full":
            print("INFO: splitting full dataset...")
            df = pd.read_feather(RAW_DATA_DIR / "Movies_and_TV.f")
        elif dataset == "5core":
            print("INFO: splitting 5 core dataset...")
            df = pd.read_feather(RAW_DATA_DIR / "Movies_and_TV_5.f")

        # df = df.sample(n=100000)

        df_rename = rename_user_item_columns(df)

        df_recent = keep_last_n_years(df_rename, n_years=5)

        df_filter = filter_users_and_items(
            df_recent, min_reviews_per_user=5, min_reviews_per_item=1
        )

        # standard time train/valid/test split
        time_train_test_split(df_filter, dataset_name=dataset)

        # leave one out train/valid/test split
        train, valid, test = leave_one_out_train_test_split(
            df_filter, dataset_name=dataset
        )
        sample_negative_test(
            train,
            valid,
            n_neg=99,
            strategy="leave_one_out",
            dataset_name=dataset,
            is_valid=True,
        )
        sample_negative_test(
            pd.concat([train, valid])
            .sort_values("user", ascending=True)
            .reset_index(drop=True),
            test,
            n_neg=99,
            strategy="leave_one_out",
            dataset_name=dataset,
            is_valid=False,
        )

        # leave N out train/valid/test split
        train, valid, test = leave_n_out_train_test_split(
            df_filter, dataset_name=dataset
        )
        sample_negative_test(
            train,
            valid,
            n_neg=99,
            strategy="leave_n_out",
            dataset_name=dataset,
            is_valid=True,
        )
        sample_negative_test(
            pd.concat([train, valid])
            .sort_values("user", ascending=True)
            .reset_index(drop=True),
            test,
            n_neg=99,
            strategy="leave_n_out",
            dataset_name=dataset,
            is_valid=False,
        )

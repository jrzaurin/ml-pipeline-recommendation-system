from collections import Counter
from functools import reduce
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
RAW_DATA_DIR = Path("data/raw/amazon")
PROCESSED_DATA_DIR = Path("data/processed/amazon")


def load_item_features(n_topics=10, n_comp=5):

    lda_fname = "_".join(["item_desc", "lda", str(n_topics) + ".f"])
    umap_fname = "_".join(["item_desc_tfidf", "umap", str(n_topics) + ".f"])

    # item features
    item_feat = pd.read_pickle(PROCESSED_DATA_DIR / "item_features.p")
    item_feat.drop(
        ["description", "title", "also_buy", "also_view"], axis=1, inplace=True
    )
    item_lda = pd.read_feather(PROCESSED_DATA_DIR / lda_fname)
    item_umap = pd.read_feather(PROCESSED_DATA_DIR / umap_fname)

    return item_feat, item_lda, item_umap


def load_interaction_data(strategy, dataset_name, is_valid):
    # Remember that here we have removed duplicates, which might be a customer
    # buying the same item. However, the truth is that when you have a look to the
    # duplicates they are mostly mistakes in the data collection. For example, out
    # of all the users used in training there are only 1511 that buy more than one
    # item in more than one diff date. I had a look to a few, and there are still
    # mistakes in data collection. In summary, we will use directly the training
    # dataset.
    if is_valid:
        tr_fname = "_".join([strategy, "tr", dataset_name + ".f"])
        train = pd.read_feather(PROCESSED_DATA_DIR / tr_fname)
    else:
        tr_fname = "_".join([strategy, "tr", dataset_name + ".f"])
        val_fname = "_".join([strategy, "val", dataset_name + ".f"])
        train = (
            pd.concat(
                [
                    pd.read_feather(PROCESSED_DATA_DIR / tr_fname),
                    pd.read_feather(PROCESSED_DATA_DIR / val_fname),
                ]
            )
            .sort_values(["user", "item"])
            .reset_index(drop=True)
        )
    return train


def time_diff(user_group):

    dates = user_group[1].reviewDate.tolist()

    if len(dates) > 1:
        delta = [(t - s).days for s, t in zip(dates, dates[1:])]
        min_time_diff, max_time_diff, mean_time_diff, median_time_diff = (
            np.min(delta),
            np.max(delta),
            np.mean(delta),
            np.median(delta),
        )
    else:
        min_time_diff, max_time_diff, mean_time_diff, median_time_diff = 0, 0, 0, 0

    return [
        user_group[0],
        min_time_diff,
        max_time_diff,
        mean_time_diff,
        median_time_diff,
    ]


def most_common_days(user_group):
    dow = user_group[1].day_of_week.mode().values[0]
    dom = user_group[1].day_of_month.mode().values[0]
    return [user_group[0], dow, dom]


def top_topics_or_umap(user_group):
    lf = np.abs(user_group[1][[c for c in user_group[1].columns if "col" in c]])
    lf = lf.values
    top_lf = np.argsort(lf, axis=1)[:, ::-1]
    top_lf = [el[0] for el in Counter(top_lf.ravel()).most_common(2)]
    return [user_group[0]] + top_lf


def time_diff_parallel(train):
    user_date_group = train.sort_values(["user", "reviewDate"]).groupby("user")[
        ["user", "reviewDate"]
    ]
    with Pool(cpu_count()) as p:
        res = p.map(time_diff, [g for g in user_date_group])
    t_diff = pd.DataFrame(
        res,
        columns=[
            "user",
            "min_time_diff",
            "max_time_diff",
            "mean_time_diff",
            "median_time_diff",
        ],
    )
    return t_diff


def most_common_days_parallel(train):
    user_date_group = train.sort_values(["user", "reviewDate"]).groupby("user")[
        ["user", "reviewDate"]
    ]
    with Pool(cpu_count()) as p:
        res = p.map(most_common_days, [g for g in user_date_group])
    day_of_week_and_month = pd.DataFrame(
        res,
        columns=[
            "user",
            "dow",
            "dom",
        ],
    )
    return day_of_week_and_month


def top_topics_parallel(train, item_lda):
    user_item_lda = pd.merge(train[["user", "item"]], item_lda, on="item")
    user_lda_group = user_item_lda.groupby("user")

    with Pool(cpu_count()) as p:
        res = p.map(top_topics_or_umap, [g for g in user_lda_group])
    top_topics = pd.DataFrame(res, columns=["user", "top_topic1", "top_topic2"])

    return top_topics


def top_umap_parallel(train, item_umap):
    user_item_umap = pd.merge(train[["user", "item"]], item_umap, on="item")
    user_umap_group = user_item_umap.groupby("user")

    with Pool(cpu_count()) as p:
        res = p.map(top_topics_or_umap, [g for g in user_umap_group])
    top_umap = pd.DataFrame(res, columns=["user", "top_umap1", "top_umap2"])

    return top_umap


def user_verified(train):
    is_user_verified = pd.concat(
        [
            pd.read_feather(PROCESSED_DATA_DIR / "train_full.f"),
            pd.read_feather(PROCESSED_DATA_DIR / "valid_full.f"),
            pd.read_feather(PROCESSED_DATA_DIR / "test_full.f"),
        ]
    )[["user", "verified"]].drop_duplicates("user", keep="last")
    is_user_verified = is_user_verified[is_user_verified.user.isin(train.user.unique())]
    return is_user_verified


def day_of_week_and_month(train):
    # I want mutate the train
    train["day_of_month"] = [d.day for d in train.reviewDate.tolist()]
    train["day_of_week"] = [d.dayofweek for d in train.reviewDate.tolist()]
    return train


def rating_stats(train):
    r_stats = (
        train.groupby("user")["overall"]
        .agg(["min", "max", "mean", "median"])
        .reset_index()
    )
    r_stats.columns = ["user"] + [
        "_".join(["ratings", c]) for c in r_stats.columns.tolist()[1:]
    ]
    return r_stats


def item_count(train):
    # number of different items bought
    i_count = train.user.value_counts().reset_index()
    i_count.columns = ["user", "item_count"]
    return i_count


def price_stats(user_item_feat):

    # item price stats
    p_stats = (
        user_item_feat.groupby("user")["price"]
        .agg(["min", "max", "mean", "median"])
        .reset_index()
    )
    p_stats.columns = ["user"] + [
        "_".join(["price", c]) for c in p_stats.columns.tolist()[1:]
    ]
    return p_stats


def most_common_value(user_item_feat, column):
    top_column_name = "_".join(["top", column])
    mcv = pd.DataFrame(
        user_item_feat.groupby("user")[column]
        .value_counts()
        .groupby("user")
        .head(1)
        .index.tolist(),
        columns=["user", column],
    )
    mcv.columns = ["user", top_column_name]
    return mcv


def muliple_merge(df_list, strategy, dataset_name, is_valid):
    if is_valid:
        out_fname = "_".join(["user_features", strategy, dataset_name, "valid.f"])
    else:
        out_fname = "_".join(["user_features", strategy, dataset_name, "test.f"])

    train_user_feat = reduce(
        lambda left, right: pd.merge(left, right, on="user", how="outer"), df_list
    )
    train_user_feat.to_feather(PROCESSED_DATA_DIR / out_fname)


def full_process(strategy, dataset_name, is_valid):

    if is_valid:
        print(
            "INFO: user feature engineering for strategy: {}, dataset: {}, and validation".format(
                strategy, dataset_name
            )
        )
    else:
        print(
            "INFO: user feature engineering for strategy: {}, dataset: {}, and test".format(
                strategy, dataset_name
            )
        )

    train = load_interaction_data(strategy, dataset_name, is_valid)
    print("INFO: number of interactions in the dataset {}".format(train.shape[0]))

    train = day_of_week_and_month(train)
    is_user_verified = user_verified(train)
    i_count = item_count(train)
    r_stats = rating_stats(train)
    t_diff = time_diff_parallel(train)
    mcd = most_common_days_parallel(train)

    item_feat, item_lda, item_umap = load_item_features()
    top_lda = top_topics_parallel(train, item_lda)
    top_umap = top_umap_parallel(train, item_umap)

    user_item_feat = pd.merge(train, item_feat, on="item")
    p_stats = price_stats(user_item_feat)
    mcc = most_common_value(user_item_feat, "category")
    mcb = most_common_value(user_item_feat, "brand")
    mcl = most_common_value(user_item_feat, "language")

    dfs = [
        is_user_verified,
        i_count,
        r_stats,
        p_stats,
        t_diff,
        mcd,
        mcc,
        mcb,
        mcl,
        top_lda,
        top_umap,
    ]

    muliple_merge(dfs, strategy, dataset_name, is_valid)


if __name__ == "__main__":

    combinations = [
        ("leave_one_out", "full", True),
        ("leave_one_out", "full", False),
        ("leave_one_out", "5core", True),
        ("leave_one_out", "5core", False),
        ("leave_n_out", "full", True),
        ("leave_n_out", "full", False),
        ("leave_n_out", "5core", True),
        ("leave_n_out", "5core", False),
    ]

    for strategy, dataset_name, is_valid in combinations:
        full_process(strategy, dataset_name, is_valid)

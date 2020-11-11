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


def time_diff_between_interactions(user_group):

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


if __name__ == "__main__":

    # Remember that here we have removed duplicates, which might be a customer
    # buying the same item. However, the truth is that when you have a look to the
    # duplicates they are mostly mistakes in the data collection. For example, out
    # of all the users used in training there are only 1511 that buy more than one
    # item in more than one diff date. I had a look to a few, and there are still
    # mistakes in data collection. In summary, we will use directly the training
    # dataset.
    train = pd.read_feather(PROCESSED_DATA_DIR / "leave_one_out_tr_full.f")
    is_user_verified = pd.concat(
        [
            pd.read_feather(PROCESSED_DATA_DIR / "train_full.f"),
            pd.read_feather(PROCESSED_DATA_DIR / "valid_full.f"),
            pd.read_feather(PROCESSED_DATA_DIR / "test_full.f"),
        ]
    )[["user", "verified"]].drop_duplicates("user", keep="last")
    is_user_verified = is_user_verified[is_user_verified.user.isin(train.user.unique())]

    # faster than apply...
    train["day_of_month"] = [d.day for d in train.reviewDate.tolist()]
    train["day_of_week"] = [d.dayofweek for d in train.reviewDate.tolist()]

    # train["score"] = train.overall * train.recency_factor

    # item features
    item_feat = pd.read_pickle(PROCESSED_DATA_DIR / "meta_movies_and_tv_processed.p")
    item_feat.drop(
        ["description", "title", "also_buy", "also_view"], axis=1, inplace=True
    )
    item_lda_10 = pd.read_feather(PROCESSED_DATA_DIR / "item_desc_lda_10.f")
    item_umap_5 = pd.read_feather(PROCESSED_DATA_DIR / "item_desc_tfidf_umap_5.f")

    # ratings stats
    ratings_stats = (
        train.groupby("user")["overall"]
        .agg(["min", "max", "mean", "median"])
        .reset_index()
    )
    ratings_stats.columns = ["user"] + [
        "_".join(["ratings", c]) for c in ratings_stats.columns.tolist()[1:]
    ]

    # time of the week and month where they buy
    user_date_group = train.sort_values(["user", "reviewDate"]).groupby("user")[
        ["user", "reviewDate"]
    ]

    with Pool(cpu_count()) as p:
        res = p.map(time_diff_between_interactions, [g for g in user_date_group])
    time_diff = pd.DataFrame(
        res,
        columns=[
            "user",
            "min_time_diff",
            "max_time_diff",
            "mean_time_diff",
            "median_time_diff",
        ],
    )

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

    # number of different items bought
    items_count = train.user.value_counts().reset_index()
    items_count.columns = ["user", "item_count"]

    # user features that depend on the items they bought
    user_item_feat = pd.merge(train, item_feat, on="item")

    # item price stats
    price_stats = (
        user_item_feat.groupby("user")["price"]
        .agg(["min", "max", "mean", "median"])
        .reset_index()
    )
    price_stats.columns = ["user"] + [
        "_".join(["price", c]) for c in price_stats.columns.tolist()[1:]
    ]

    # user most common category
    most_common_cat = pd.DataFrame(
        user_item_feat.groupby("user")["category"]
        .value_counts()
        .groupby("user")
        .head(1)
        .index.tolist(),
        columns=["user", "category"],
    )
    most_common_cat.columns = ["user", "top_category"]

    # user most common brand
    most_common_brand = pd.DataFrame(
        user_item_feat.groupby("user")["brand"]
        .value_counts()
        .groupby("user")
        .head(1)
        .index.tolist(),
        columns=["user", "brand"],
    )
    most_common_brand.columns = ["user", "top_brand"]

    # user most common language
    most_common_lang = pd.DataFrame(
        user_item_feat.groupby("user")["language"]
        .value_counts()
        .groupby("user")
        .head(1)
        .index.tolist(),
        columns=["user", "language"],
    )
    most_common_lang.columns = ["user", "top_language"]

    # user top 2 topics and UMAP dims
    user_item_lda = pd.merge(train[["user", "item"]], item_lda_10, on="item")
    user_lda_group = user_item_lda.groupby("user")
    user_item_umap = pd.merge(train[["user", "item"]], item_umap_5, on="item")
    user_umap_group = user_item_umap.groupby("user")

    with Pool(cpu_count()) as p:
        res = p.map(top_topics_or_umap, [g for g in user_lda_group])
    top_topics = pd.DataFrame(res, columns=["user", "top_topic1", "top_topic2"])

    with Pool(cpu_count()) as p:
        res = p.map(top_topics_or_umap, [g for g in user_umap_group])
    top_umap = pd.DataFrame(res, columns=["user", "top_umap1", "top_umap2"])

    # merge all
    dfs = [
        is_user_verified,
        ratings_stats,
        time_diff,
        day_of_week_and_month,
        items_count,
        price_stats,
        most_common_cat,
        most_common_brand,
        most_common_lang,
        top_topics,
        top_umap,
    ]
    train_user_feat = reduce(
        lambda left, right: pd.merge(left, right, on="user", how="outer"), dfs
    )
    train_user_feat.to_feather(PROCESSED_DATA_DIR / "train_user_feat.f")

import pickle
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from label_encoder import LabelEncoder
from split_data import compute_recency_factor

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

PROCESSED_DATA_DIR = Path("data/processed/amazon")
RESULTS_DIR = Path("results")


def nan_with_unknown_imputer(df, columns):
    for c in columns:
        df[c] = df[c].fillna("unknown")
    return df


def nan_with_minus_one_imputer(df, columns):
    for c in columns:
        df[c] = df[c].fillna(-1.0).astype("float")
    return df


def get_train_instances(train_arr, train_pos, test_w_neg, n_items, n_neg):

    user, item, labels = [], [], []
    for u, i, r in tqdm(train_arr):

        # we need to make sure they are not in the negative examples used for
        # testing
        try:
            user_test_neg = test_w_neg[u]
        except KeyError:
            user_test_neg = [-666]

        for _ in range(n_neg):
            j = np.random.randint(n_items)
            while j in train_pos[u] or j in user_test_neg:
                j = np.random.randint(n_items)
            user.append(u)
            item.append(j)
            labels.append(0)

    train_w_negative = np.vstack([user, item, labels]).T

    return train_w_negative.astype(np.int64)


def load_interaction_data(strategy, dataset_name, is_valid):
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


def load_user_and_item_idx(strategy, dataset_name, is_valid):

    if is_valid:
        item_fname = "_".join(
            ["item_idx", strategy, "w_negative", dataset_name, "valid.p"]
        )
        user_fname = "_".join(
            ["user_idx", strategy, "w_negative", dataset_name, "valid.p"]
        )
    else:
        item_fname = "_".join(
            ["item_idx", strategy, "w_negative", dataset_name, "test.p"]
        )
        user_fname = "_".join(
            ["user_idx", strategy, "w_negative", dataset_name, "test.p"]
        )

    item_idx = pd.read_pickle(PROCESSED_DATA_DIR / item_fname)
    user_idx = pd.read_pickle(PROCESSED_DATA_DIR / user_fname)

    return item_idx, user_idx


def load_item_feat(item_idx, n_topics=10, n_umap=5):

    lda_fname = "_".join(["item_desc_lda", str(n_topics) + ".f"])
    umap_fname = "_".join(["item_desc_tfidf_umap", str(n_umap) + ".f"])

    # ITEM FEATURES
    item_feat = pd.read_pickle(PROCESSED_DATA_DIR / "item_features.p")[
        ["item", "brand", "price", "category", "language"]
    ]

    item_umap = pd.read_feather(PROCESSED_DATA_DIR / umap_fname)
    item_lda = pd.read_feather(PROCESSED_DATA_DIR / lda_fname)

    # rename umap and lda features. This should have been done earlier in the
    # process
    umap_cols, lda_cols = [], []
    for i, c in enumerate(item_umap.columns):
        if "col" in c:
            umap_cols.append("umap_" + str(i))
        else:
            umap_cols.append(c)

    for i, c in enumerate(item_lda.columns):
        if "col" in c:
            lda_cols.append("lda_" + str(i))
        else:
            lda_cols.append(c)

    item_umap.columns = umap_cols
    item_lda.columns = lda_cols

    item_feat = nan_with_unknown_imputer(
        item_feat, columns=["brand", "category", "language"]
    )
    item_feat = nan_with_minus_one_imputer(item_feat, columns=["price"])

    item_feat = item_feat[item_feat.item.isin(item_idx.keys())]
    item_umap = item_umap[item_umap.item.isin(item_idx.keys())]
    item_lda = item_lda[item_lda.item.isin(item_idx.keys())]
    dfs = [
        item_feat,
        item_umap,
        item_lda,
    ]
    item_feat = reduce(lambda left, right: pd.merge(left, right, on="item"), dfs)
    item_feat["item"] = item_feat["item"].map(item_idx).astype(int)

    return item_feat


def load_user_feat(user_idx, strategy, dataset_name, is_valid):

    if is_valid:
        user_feat_fname = "_".join(["user_features", strategy, dataset_name, "valid.f"])
    else:
        user_feat_fname = "_".join(["user_features", strategy, dataset_name, "test.f"])

    # USER FEATURES
    user_feat = pd.read_feather(PROCESSED_DATA_DIR / user_feat_fname)
    user_feat = nan_with_unknown_imputer(
        user_feat,
        columns=[
            c
            for c in user_feat.columns
            if c
            in ["verified", "dow", "dom", "top_brand", "top_category", "top_language"]
        ],
    )
    user_feat = nan_with_minus_one_imputer(
        user_feat,
        columns=[
            c
            for c in user_feat.columns
            if c
            not in [
                "user",
                "verified",
                "dow",
                "dom",
                "top_brand",
                "top_category",
                "top_language",
            ]
        ],
    )

    user_feat["user"] = user_feat["user"].map(user_idx).astype(int)

    return user_feat


def rating_w_recency(train, strategy, dataset_name, is_valid):
    tr_interact = load_interaction_data(strategy, dataset_name, is_valid)
    # a patch for a "bad design in the split earlier
    if tr_interact.recency_factor.isna().sum() > 0:
        compute_recency_factor(tr_interact, xmid=730, tau=120, top=1)
    tr_interact["user"] = tr_interact["user"].map(user_idx).astype(int)
    tr_interact["item"] = tr_interact["item"].map(item_idx).astype(int)
    tr_interact["rating"] = tr_interact.overall * tr_interact.recency_factor
    train = train.merge(
        tr_interact[["user", "item", "rating", "recency_factor"]], on=["user", "item"]
    )
    assert np.all((train.rating_y == train.rating_x * train.recency_factor).values)
    train.drop(["rating_x", "recency_factor"], axis=1, inplace=True)
    train.rename(columns={"rating_y": "rating"}, inplace=True)
    return train


def sample_negative_train(
    strategy,
    is_valid,
    dataset_name,
    item_feat,
    user_feat,
    item_idx,
    user_idx,
    with_recency,
):

    if is_valid:
        data_fname = "_".join([strategy, "w_negative", dataset_name, "valid.npz"])
        item_pop_fname = "_".join(
            ["item_popularity", strategy, dataset_name, "valid.f"]
        )
    else:
        data_fname = "_".join([strategy, "w_negative", dataset_name, "test.npz"])
        item_pop_fname = "_".join(["item_popularity", strategy, dataset_name, "test.f"])

    train_test_set = np.load(PROCESSED_DATA_DIR / data_fname)

    # sampling negatives: building dictionaries since is way faster
    train = pd.DataFrame(train_test_set["train"], columns=["user", "item", "rating"])

    if with_recency:
        train = rating_w_recency(train, strategy, dataset_name, is_valid)

    train_lookup = train.groupby("user")["item"].apply(list)
    test_w_neg = pd.DataFrame(
        train_test_set["test"], columns=["user", "item", "rating"]
    )
    test_lookup = test_w_neg.groupby("user")["item"].apply(list)
    train_neg = get_train_instances(
        train_test_set["train"],
        train_lookup,
        test_lookup,
        train_test_set["n_items"],
        n_neg=4,
    )

    # training set with negatives
    train_w_neg = (
        pd.DataFrame(
            np.vstack([train_test_set["train"], train_neg]),
            columns=["user", "item", "rating"],
        )
        .sort_values(["user", "item"])
        .drop_duplicates(["user", "item"])
        .reset_index(drop=True)
    )

    # lets remove those items without metadata from the training and testing
    # set
    train_w_neg = train_w_neg[train_w_neg.item.isin(item_feat.item)]
    test_w_neg = test_w_neg[test_w_neg.item.isin(item_feat.item)]

    # merge interactions with user and item features
    item_pop = pd.read_feather(PROCESSED_DATA_DIR / item_pop_fname)
    item_pop["item"] = item_pop.item.map(item_idx)
    item_feat = pd.merge(item_feat, item_pop, on="item").reset_index(drop=True)
    train_w_neg = pd.merge(
        pd.merge(train_w_neg, item_feat, on="item"), user_feat, on="user"
    )
    test_w_neg = pd.merge(
        pd.merge(test_w_neg, item_feat, on="item"), user_feat, on="user"
    )

    return train_w_neg, test_w_neg


def encode(train_w_neg, test_w_neg, strategy, dataset_name, is_valid, with_recency):

    if with_recency:
        dataset_name = "_".join(["recency", dataset_name])
    if is_valid:
        encoder_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "valid", "encoder.p"]
        )
        tr_out_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "train_valid", "le.f"]
        )
        te_out_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "valid", "le.f"]
        )
    else:
        encoder_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "test", "encoder.p"]
        )
        tr_out_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "train_test", "le.f"]
        )
        te_out_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "test", "le.f"]
        )

    # cols that will be treated as categorical in training have to be int
    int_cols = [
        "verified",
        "dow",
        "dom",
        "top_topic1",
        "top_topic2",
        "top_umap1",
        "top_umap2",
    ]
    train_w_neg[int_cols] = train_w_neg[int_cols].astype(int)
    test_w_neg[int_cols] = test_w_neg[int_cols].astype(int)

    # "proper" categorical columsn
    categ_cols = [c for c in train_w_neg.columns if train_w_neg[c].dtype == "O"]

    # Label Encode and save
    encoder = LabelEncoder(categ_cols)
    train_w_neg_le = encoder.fit_transform(train_w_neg)
    test_w_neg_le = encoder.transform(test_w_neg)

    train_w_neg_le.to_feather(PROCESSED_DATA_DIR / tr_out_fname)
    test_w_neg_le.to_feather(PROCESSED_DATA_DIR / te_out_fname)
    pickle.dump(encoder, open(PROCESSED_DATA_DIR / encoder_fname, "wb"))


if __name__ == "__main__":

    # for the time being going to leave the 5core dataset and leave_n_out out
    # the GBM experiment
    combinations = [
        # strategy, dataset_name, is_valid, with_recency
        ("leave_one_out", "full", True, False),
        ("leave_one_out", "full", True, True),
        ("leave_one_out", "full", False, False),
        ("leave_one_out", "full", False, True),
    ]

    for strategy, dataset_name, is_valid, with_recency in combinations:
        print(
            "INFO: processing data for the gbm with strategy: {}, for dataset: {}, validation: {}, and recency: {}".format(
                strategy, dataset_name, str(is_valid), str(with_recency)
            )
        )

        item_idx, user_idx = load_user_and_item_idx(strategy, dataset_name, is_valid)

        item_feat = load_item_feat(item_idx)

        user_feat = load_user_feat(user_idx, strategy, dataset_name, is_valid)

        train_w_neg, test_w_neg = sample_negative_train(
            strategy,
            is_valid,
            dataset_name,
            item_feat,
            user_feat,
            item_idx,
            user_idx,
            with_recency,
        )

        encode(train_w_neg, test_w_neg, strategy, dataset_name, is_valid, with_recency)

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from label_encoder import LabelEncoder

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


# item features
item_feat = pd.read_pickle(PROCESSED_DATA_DIR / "meta_movies_and_tv_processed.p")[
    ["item", "brand", "price", "category", "language"]
]

item_umap_5 = pd.read_feather(PROCESSED_DATA_DIR / "item_desc_tfidf_umap_5.f")
item_lda_10 = pd.read_feather(PROCESSED_DATA_DIR / "item_desc_lda_10.f")

umap_cols, lda_cols = [], []
for i, c in enumerate(item_umap_5.columns):
    if 'col' in c:
        umap_cols.append("umap_" + str(i))
    else:
        umap_cols.append(c)

for i, c in enumerate(item_lda_10.columns):
    if 'col' in c:
        lda_cols.append("lda_" + str(i))
    else:
        lda_cols.append(c)

item_umap_5.columns = umap_cols
item_lda_10.columns = lda_cols

item_feat = nan_with_unknown_imputer(
    item_feat, columns=["brand", "category", "language"]
)
item_feat = nan_with_minus_one_imputer(item_feat, columns=["price"])

# user features
user_feat = pd.read_feather(PROCESSED_DATA_DIR / "train_user_feat.f")
user_feat = nan_with_unknown_imputer(
    user_feat,
    columns=[
        c
        for c in user_feat.columns
        if c in ["verified", "dow", "dom", "top_brand", "top_category", "top_language"]
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

# mapping dictionaries
item_idx = pd.read_pickle(
    PROCESSED_DATA_DIR / "item_idx_leave_one_out_w_negative_full_valid.p"
)
user_idx = pd.read_pickle(
    PROCESSED_DATA_DIR / "user_idx_leave_one_out_w_negative_full_valid.p"
)

# training and validation set (test here is validation)
train_test_set = np.load(PROCESSED_DATA_DIR / "leave_one_out_w_negative_full_valid.npz")

# sampling negatives
train = pd.DataFrame(train_test_set["train"], columns=["user", "item", "rating"])
train_lookup = train.groupby("user")["item"].apply(list)
test_w_neg = pd.DataFrame(train_test_set["test"], columns=["user", "item", "rating"])
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


user_feat["user"] = user_feat["user"].map(user_idx).astype(int)

item_feat = item_feat[item_feat.item.isin(item_idx.keys())]
# there are 167 items for which there is not metadata (i.e. len(item_idx) - item_feat.shape[0] = 167)
item_umap_5 = item_umap_5[item_umap_5.item.isin(item_idx.keys())]
item_lda_10 = item_lda_10[item_lda_10.item.isin(item_idx.keys())]
dfs = [
    item_feat,
    item_umap_5,
    item_lda_10,
]
item_feat = reduce(
    lambda left, right: pd.merge(left, right, on="item"), dfs
)
item_feat["item"] = item_feat["item"].map(item_idx).astype(int)

# lets remove those 167 from the training and testing set
train_w_neg = train_w_neg[train_w_neg.item.isin(item_feat.item)]
test_w_neg = test_w_neg[test_w_neg.item.isin(item_feat.item)]

# merge
train_w_neg = pd.merge(
    pd.merge(train_w_neg, item_feat, on="item"), user_feat, on="user"
)
test_w_neg = pd.merge(pd.merge(test_w_neg, item_feat, on="item"), user_feat, on="user")

categ_cols = [c for c in train_w_neg.columns if train_w_neg[c].dtype == "O"] + [
    "verified",
    "dow",
    "dom",
    "top_topic1",
    "top_topic2",
    "top_umap1",
    "top_umap2",
]
encoder = LabelEncoder(categ_cols)
train_w_neg_le = encoder.fit_transform(train_w_neg)
test_w_neg_le = encoder.transform(test_w_neg)

import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

PROCESSED_DATA_DIR = Path("data/processed/amazon")

# 1. load train and valid, interactions_mtx and idx dicts
train = pd.read_feather(PROCESSED_DATA_DIR / "train.f")
train = train[
    train.reviewerID.isin(
        train.reviewerID.value_counts()[
            train.reviewerID.value_counts() >= 5
        ].index.tolist()
    )
]
valid = pd.read_feather(PROCESSED_DATA_DIR / "valid.f")
interactions_mtx_binary = load_npz(PROCESSED_DATA_DIR / "interactions_mtx_binary.npz")
# interactions_mtx_score = load_npz(PROCESSED_DATA_DIR / "interactions_mtx_score.npz")
items_idx = pickle.load(open(PROCESSED_DATA_DIR / "items_idx.p", "rb"))
users_idx = pickle.load(open(PROCESSED_DATA_DIR / "users_idx.p", "rb"))

# 2. when using knn-cf, we can only recommend to users and items seen in
# training, so filter accordingly
valid_hot = valid[
    (valid.asin.isin(train.asin)) & (valid.reviewerID.isin(train.reviewerID))
]

# 3 Most popular items
most_popular_items = train.asin.value_counts().reset_index()
most_popular_items.columns = ["asin", "counts"]
most_popular_items["item"] = [items_idx[k] for k in most_popular_items.asin.tolist()]

# 4. to indexes
train["user"] = [users_idx[k] for k in train.reviewerID.tolist()]
train["item"] = [items_idx[k] for k in train.asin.tolist()]
valid_hot["user"] = [users_idx[k] for k in valid_hot.reviewerID.tolist()]
valid_hot["item"] = [items_idx[k] for k in valid_hot.asin.tolist()]

# 5. train knn-item-based-cf
k = 100
interactions_mtx_binary_knn = interactions_mtx_binary.T
model = NearestNeighbors(metric="cosine", n_neighbors=5)
model.fit(interactions_mtx_binary_knn)


def recall_binary(rec, true, k):
    tp = np.intersect1d(rec[:k], true).shape[0]
    allp = true.shape[0]
    return tp / allp


def hit_ratio(rec, true, k):
    return np.intersect1d(rec[:k], true).shape[0]


def _dcg_binary(rec, true, k):
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    _, rank, _ = np.intersect1d(rec[:k], true, return_indices=True)
    dcg = discount[rank].sum()
    return dcg


def ndgc_binary(rec, true, k):
    dcg = _dcg_binary(rec, true, k)
    idcg = _dcg_binary(true, true, k)
    return dcg / idcg


def _dcg_score(rec, true, true_score, k):
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    rec_item, rec_rank, true_loc = np.intersect1d(rec[:k], true, return_indices=True)
    true_relevance = true_score[true_loc]
    discount = discount[rec_rank]
    dcg = true_relevance * discount
    return dcg


def ndgc_score(rec, true, true_relevance, k):
    dcg = _dcg_binary(rec, true, true_relevance, k)
    idcg = _dcg_binary(true, true, true_relevance, k)
    return dcg / idcg


def knn_item_cf(group):

    # given a user seen during training and validation, get her training interactions
    user = group[0]
    movies = group[1].item.values

    # compute the k NN
    dist, nnidx = model.kneighbors(interactions_mtx_binary_knn[movies], n_neighbors=k + 1)

    # Drop the 1st result as the closest to a movie is always itself
    dist, nnidx = dist[:, 1:], nnidx[:, 1:]
    dist, nnidx = dist.flatten(), nnidx.flatten()

    # rank based on distances and keep top k
    recs = nnidx[np.argsort(dist)][:k]

    return (user, recs)


users_sample = valid_hot.user.sample(50000).unique()
user_groups = train[train.user.isin(users_sample)].groupby("user")


results: dict = {}
results['knn_cf'] = {}
results['most_popular'] = {}
with Pool(cpu_count()) as p:
    res = p.map(knn_item_cf, [g for g in user_groups])

ndgc_knn, rec_knn, hr_knn = [], [], []
ndgc_mp, rec_mp, hr_mp = [], [], []
mp_recs = most_popular_items.item.values[:k]
for recs in res:
    true = valid_hot[valid_hot.user == recs[0]].item.values
    knn_recs = recs[1]

    ndgc_knn.append(ndgc_binary(knn_recs, true, k))
    rec_knn.append(recall_binary(knn_recs, true, k))
    hr_knn.append(hit_ratio(knn_recs, true, k))

    ndgc_mp.append(ndgc_binary(mp_recs, true, k))
    rec_mp.append(recall_binary(mp_recs, true, k))
    hr_mp.append(hit_ratio(mp_recs, true, k))

results['knn_cf']["ndgc_knn"] = np.mean(ndgc_knn)
results['knn_cf']["rec_knn"] = np.mean(rec_knn)
results['knn_cf']["hr_knn"] = np.mean(hr_knn)
results['most_popular']["ndgc_mp"] = np.mean(ndgc_mp)
results['most_popular']["rec_mp"] = np.mean(rec_mp)
results['most_popular']["hr_mp"] = np.mean(hr_mp)

pickle.dump(results, open(PROCESSED_DATA_DIR / "knn_item_cf_results.p", "wb"))

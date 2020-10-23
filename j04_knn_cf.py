import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

PROCESSED_DATA_DIR = Path("data/processed/amazon")

# 1. load train and valid, interactions_mtx and idx dicts
train = pd.read_feather(PROCESSED_DATA_DIR / "train.f")
valid = pd.read_feather(PROCESSED_DATA_DIR / "valid.f")
interactions_mtx_binary = load_npz(PROCESSED_DATA_DIR / "interactions_mtx_binary.npz")
interactions_mtx_score = load_npz(PROCESSED_DATA_DIR / "interactions_mtx_score.npz")
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
interactions_mtx_binary_knn = interactions_mtx_binary.T
# interactions_mtx_score_knn = interactions_mtx_score.T

model1_binary = NearestNeighbors(n_neighbors=10)
# model1_score = NearestNeighbors(n_neighbors=10)
model2_binary = NearestNeighbors(algorithm="brute", metric="cosine", n_neighbors=10)
# model2_score = NearestNeighbors(algorithm="brute", metric="cosine", n_neighbors=10)

model1_binary.fit(interactions_mtx_binary_knn)
# model1_score.fit(interactions_mtx_score_knn)
model2_binary.fit(interactions_mtx_binary_knn)
# model2_score.fit(interactions_mtx_score_knn)


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


users_sample = valid_hot.user.sample(100).unique()
user_groups = train[train.user.isin(users_sample)].groupby("user")
valid_hot_sample = valid_hot[valid_hot.user.isin(users_sample)]


def knn_item_cf(group, model, interactions_mtx, k):

    # given a user seen during training and validation, get her training interactions
    user = group[0]
    movies = group[1].item.values

    # compute the k NN
    dist, nnidx = model.kneighbors(interactions_mtx[movies], n_neighbors=k + 1)

    # Drop the 1st result as the closest to a movie is always itself
    dist, nnidx = dist[:, 1:], nnidx[:, 1:]
    dist, nnidx = dist.flatten(), nnidx.flatten()

    # rank based on distances and keep top k
    recs = nnidx[np.argsort(dist)][:k]

    return (user, recs)


experiments = [
    ("model1", model1_binary, interactions_mtx_binary_knn, 20),
    ("model1", model1_binary, interactions_mtx_binary_knn, 50),
    ("model1", model1_binary, interactions_mtx_binary_knn, 100),
    ("model2", model2_binary, interactions_mtx_binary_knn, 20),
    ("model2", model2_binary, interactions_mtx_binary_knn, 50),
    ("model2", model2_binary, interactions_mtx_binary_knn, 100),
]

results: dict = {}
for name, model, interactions_mtx, k in experiments:

    print("INFO: model: {}, k: {}".format(name, k))

    start = time()
    exp_name = "_".join([name, str(k)])
    results[exp_name] = {}

    with Pool(cpu_count()) as p:
        res = p.map(
            partial(knn_item_cf, model=model, interactions_mtx=interactions_mtx, k=k),
            [g for g in user_groups],
        )

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

    results[exp_name]["ndgc_knn"] = np.mean(ndgc_knn)
    results[exp_name]["rec_knn"] = np.mean(rec_knn)
    results[exp_name]["hr_knn"] = np.mean(hr_knn)
    results[exp_name]["ndgc_mp"] = np.mean(ndgc_mp)
    results[exp_name]["rec_mp"] = np.mean(rec_mp)
    results[exp_name]["hr_mp"] = np.mean(hr_mp)
    end = time() - start

    print("INFO: model: {}, k: {}, running time: {}".format(name, k, end // 60))


pickle.dump(results, open(PROCESSED_DATA_DIR / "knn_item_cf_results.p", "wb"))

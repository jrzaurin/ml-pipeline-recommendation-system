import os
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

from metrics import hit_ratio, ndgc_binary, recall_binary

PROCESSED_DATA_DIR = Path("data/processed/amazon")
RESULTS_DIR = Path("results")


def knn_item_cf_recs(group, model, interactions_mtx, k):

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


def run_experiments(
    dataset,
    min_reviews_per_user,
    k=100,
    sample=None,
):

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    results_fname = "_".join(
        ["knn_item_cf_results", dataset, str(min_reviews_per_user), "k", str(k) + ".p"]
    )

    train = pd.read_feather(PROCESSED_DATA_DIR / "_".join(["train", dataset + ".f"]))
    valid = pd.read_feather(PROCESSED_DATA_DIR / "_".join(["valid", dataset + ".f"]))
    interactions_mtx = load_npz(
        PROCESSED_DATA_DIR
        / "_".join(["interactions_mtx", dataset, str(min_reviews_per_user) + ".npz"])
    )
    items_idx = pickle.load(
        open(
            PROCESSED_DATA_DIR
            / "_".join(["items_idx", dataset, str(min_reviews_per_user) + ".p"]),
            "rb",
        )
    )
    users_idx = pickle.load(
        open(
            PROCESSED_DATA_DIR
            / "_".join(["users_idx", dataset, str(min_reviews_per_user) + ".p"]),
            "rb",
        )
    )

    train = train[
        train.reviewerID.isin(
            train.reviewerID.value_counts()[
                train.reviewerID.value_counts() >= min_reviews_per_user
            ].index.tolist()
        )
    ].reset_index(drop=True)

    # when using knn-cf, we can only recommend to users and items seen in
    # training, so filter accordingly
    valid_hot = valid[
        (valid.asin.isin(train.asin)) & (valid.reviewerID.isin(train.reviewerID))
    ].reset_index(drop=True)

    # extract most popular items
    most_popular_items = train.asin.value_counts().reset_index()
    most_popular_items.columns = ["asin", "counts"]
    most_popular_items["item"] = [
        items_idx[k] for k in most_popular_items.asin.tolist()
    ]

    # all to indexes
    train["user"] = [users_idx[k] for k in train.reviewerID.tolist()]
    train["item"] = [items_idx[k] for k in train.asin.tolist()]
    valid_hot["user"] = [users_idx[k] for k in valid_hot.reviewerID.tolist()]
    valid_hot["item"] = [items_idx[k] for k in valid_hot.asin.tolist()]

    # train knn-item-based-cf
    interactions_mtx_knn = interactions_mtx.T
    model = NearestNeighbors(metric="cosine", n_neighbors=5)
    model.fit(interactions_mtx_knn)

    if sample is not None:
        valid_users_sample = valid_hot.user.sample(sample).unique()
        user_groups = train[train.user.isin(valid_users_sample)].groupby("user")
    else:
        valid_users = valid_hot.user.unique()
        user_groups = train[train.user.isin(valid_users)].groupby("user")

    print("number of users used for this experiment: {}.".format(len(user_groups)))
    print("running knn cf...")
    start = time()
    with Pool(cpu_count()) as p:
        res = p.map(
            partial(
                knn_item_cf_recs,
                model=model,
                interactions_mtx=interactions_mtx_knn,
                k=k,
            ),
            [g for g in user_groups],
        )
    end = time() - start
    print("knn cf run in {} sec".format(round(end, 3)))

    results: dict = {}
    results["knn_cf"] = {}
    results["most_popular"] = {}
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

    results["knn_cf"]["ndgc_knn"] = np.mean(ndgc_knn)
    results["knn_cf"]["rec_knn"] = np.mean(rec_knn)
    results["knn_cf"]["hr_knn"] = np.mean(hr_knn)
    results["most_popular"]["ndgc_mp"] = np.mean(ndgc_mp)
    results["most_popular"]["rec_mp"] = np.mean(rec_mp)
    results["most_popular"]["hr_mp"] = np.mean(hr_mp)

    pickle.dump(results, open(RESULTS_DIR / results_fname, "wb"))


if __name__ == "__main__":

    run_experiments(dataset="full", min_reviews_per_user=3, k=20)
    run_experiments(dataset="full", min_reviews_per_user=3, k=50)
    run_experiments(dataset="full", min_reviews_per_user=3, k=100)
    run_experiments(dataset="full", min_reviews_per_user=5, k=20)
    run_experiments(dataset="full", min_reviews_per_user=5, k=50)
    run_experiments(dataset="full", min_reviews_per_user=5, k=100)
    run_experiments(dataset="full", min_reviews_per_user=7, k=20)
    run_experiments(dataset="full", min_reviews_per_user=7, k=50)
    run_experiments(dataset="full", min_reviews_per_user=7, k=100)

    run_experiments(dataset="5core", min_reviews_per_user=3, k=20)
    run_experiments(dataset="5core", min_reviews_per_user=3, k=50)
    run_experiments(dataset="5core", min_reviews_per_user=3, k=100)
    run_experiments(dataset="5core", min_reviews_per_user=5, k=20)
    run_experiments(dataset="5core", min_reviews_per_user=5, k=50)
    run_experiments(dataset="5core", min_reviews_per_user=5, k=100)
    run_experiments(dataset="5core", min_reviews_per_user=7, k=20)
    run_experiments(dataset="5core", min_reviews_per_user=7, k=50)
    run_experiments(dataset="5core", min_reviews_per_user=7, k=100)

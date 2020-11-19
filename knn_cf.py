import os
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from metrics import hit_ratio, ndcg_binary

PROCESSED_DATA_DIR = Path("data/processed/amazon")
RESULTS_DIR = Path("results")
IS_VALID = False


def knn_item_cf_recs(te_lookup, tr_lookup, urm, agg_method="mean"):

    # get train and valid interactions
    user = te_lookup[0]
    te_items = te_lookup[1]
    tr_items = tr_lookup[user]

    # compute cosine distance between training and the 100 val itens (1 pos +
    # 99 neg)
    dist = pairwise_distances(urm[tr_items], urm[te_items], metric="cosine")

    # dist will be a (N_items_tr, 100) matrix. We aggregate over rows -> (1,
    # 100) and then we sort
    if agg_method == "min":
        idx = np.argsort(np.min(dist, axis=0))
    elif agg_method == "mean":
        idx = np.argsort(np.mean(dist, axis=0))
    else:
        print("Only 'min' and 'mean' are allowed as 'agg_method'")

    recs = np.array(te_items)[idx]
    return (user, recs)


def get_recommendations(
    dataset,
    strategy,
    sample=None,
):

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if IS_VALID:
        input_fname = "_".join([strategy, "w_negative", dataset, "valid.npz"])
        URM_fname = "_".join(["URM", strategy, dataset, "valid.npz"])
        train = pd.DataFrame(
            np.load(PROCESSED_DATA_DIR / input_fname)["train"],
            columns=["user", "item", "rating"],
        )
        test = pd.DataFrame(
            np.load(PROCESSED_DATA_DIR / input_fname, allow_pickle=True)["test"],
            columns=["user", "item", "rating"],
        )
        urm = load_npz(PROCESSED_DATA_DIR / URM_fname)
    else:
        input_fname = "_".join([strategy, "w_negative", dataset, "test.npz"])
        URM_fname = "_".join(["URM", strategy, dataset, "test.npz"])
        train = pd.DataFrame(
            np.load(PROCESSED_DATA_DIR / input_fname)["train"],
            columns=["user", "item", "rating"],
        )
        test = pd.DataFrame(
            np.load(PROCESSED_DATA_DIR / input_fname)["test"],
            columns=["user", "item", "rating"],
        )
        urm = load_npz(PROCESSED_DATA_DIR / URM_fname)

    # extract most popular items. MP items will also be ranked based on
    # similarity to the training items for each user
    most_popular_items = train.item.value_counts().reset_index()
    most_popular_items.columns = ["item", "counts"]
    mp_recs = most_popular_items.item.values[:100]

    if sample is not None:
        test_users_sample = test.user.sample(sample, random_state=1).unique()
        test_lookup = (
            test[test.user.isin(test_users_sample)].groupby("user")["item"].apply(list)
        )
        train_lookup = (
            train[train.user.isin(test_users_sample)]
            .groupby("user")["item"]
            .apply(list)
        )
        test_pos_lookup = (
            test[(test.user.isin(test_users_sample)) & (test.rating != 0)]
            .groupby("user")["item"]
            .apply(list)
        )

    else:
        train_lookup = train.groupby("user")["item"].apply(list)
        test_lookup = test.groupby("user")["item"].apply(list)
        test_pos_lookup = test[test.rating != 0].groupby("user")["item"].apply(list)

    print("number of users used for this experiment: {}.".format(len(test_lookup)))
    print("running knn cf...")
    start = time()
    with Pool(cpu_count()) as p:
        knn_recs = p.map(
            partial(knn_item_cf_recs, tr_lookup=train_lookup, urm=urm),
            [tp for tp in test_lookup.items()],
        )
    end = time() - start
    print("knn cf run in {} sec".format(round(end, 3)))

    return test_pos_lookup, mp_recs, knn_recs


def run_experiments(dataset, strategy, k, sample=None):

    if IS_VALID:
        results_fname = "_".join(
            ["knn_item_cf_results", strategy, dataset, "valid", "k", str(k) + ".p"]
        )
    else:
        results_fname = "_".join(
            ["knn_item_cf_results", strategy, dataset, "test", "k", str(k) + ".p"]
        )

    test_pos, mp_recs, knn_recs = get_recommendations(dataset, strategy, sample=sample)
    results: dict = {}
    results["knn_cf"] = {}
    results["most_popular"] = {}
    ndgc_knn, hr_knn = [], []
    ndgc_mp, hr_mp = [], []

    for recs in tqdm(knn_recs):
        true = test_pos[recs[0]]
        mp_rec = mp_recs[:k]
        rec = recs[1][:k]
        ndgc_knn.append(ndcg_binary(rec, true, k))
        hr_knn.append(hit_ratio(rec, true, k))
        ndgc_mp.append(ndcg_binary(mp_rec, true, k))
        hr_mp.append(hit_ratio(mp_rec, true, k))

    results["knn_cf"]["ndgc"] = np.mean(ndgc_knn)
    results["knn_cf"]["hr"] = np.mean(hr_knn)
    results["most_popular"]["ndgc"] = np.mean(ndgc_mp)
    results["most_popular"]["hr"] = np.mean(hr_mp)

    pickle.dump(results, open(RESULTS_DIR / results_fname, "wb"))


if __name__ == "__main__":

    run_experiments(dataset="full", strategy="leave_one_out", k=10)
    run_experiments(dataset="full", strategy="leave_one_out", k=20)
    run_experiments(dataset="full", strategy="leave_one_out", k=50)
    # run_experiments(dataset="full", strategy="leave_n_out", k=10)
    # run_experiments(dataset="full", strategy="leave_n_out", k=20)
    # run_experiments(dataset="full", strategy="leave_n_out", k=50)

    # run_experiments(dataset="5core", strategy="leave_one_out", k=10)
    # run_experiments(dataset="5core", strategy="leave_one_out", k=20)
    # run_experiments(dataset="5core", strategy="leave_one_out", k=50)
    # run_experiments(dataset="5core", strategy="leave_n_out", k=10)
    # run_experiments(dataset="5core", strategy="leave_n_out", k=20)
    # run_experiments(dataset="5core", strategy="leave_n_out", k=50)

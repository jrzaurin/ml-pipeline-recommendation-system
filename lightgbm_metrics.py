# import warnings
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import hit_ratio, ndcg_binary

# from tqdm import tqdm

# warnings.filterwarnings("error")
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

PROCESSED_DATA_DIR = Path("data/processed/amazon")
RESULTS_DIR = Path("results")


def get_metrics(group, k):
    df = group[1]
    true = df[df.score != 0]["item"].values
    rec = df.sort_values("preds", ascending=False)["item"].values[:k]
    return (hit_ratio(rec, true, k), ndcg_binary(rec, true, k))


def experiment_metrics(strategy, dataset_name, is_valid, binary, multiclass, recency):

    if is_valid:
        dataset_name = "_".join([dataset_name, "valid"])
    else:
        dataset_name = "_".join([dataset_name, "test"])

    if binary:
        dataset_name = "_".join([dataset_name, "binary"])

    if multiclass:
        dataset_name = "_".join([dataset_name, "multiclass"])

    if recency:
        dataset_name = "_".join([dataset_name, "recency"])

    preds_fname = "_".join(["lgb", strategy, dataset_name, "preds.f"])

    eval_preds = pd.read_feather(PROCESSED_DATA_DIR / preds_fname)
    users_score = eval_preds.groupby("user")["score"].sum().reset_index()
    users_to_drop = users_score[users_score.score == 0].user.values
    eval_preds = eval_preds[~eval_preds.user.isin(users_to_drop)]
    user_groups = eval_preds.groupby("user")

    with Pool(cpu_count()) as p:
        res = p.map(partial(get_metrics, k=10), [g for g in user_groups])

    hr = np.mean([el[0] for el in res])
    ndgc = np.mean([el[1] for el in res])

    return hr, ndgc


def save_results(
    hr, ndgc, strategy, dataset_name, is_valid, binary, multiclass, recency
):

    if is_valid:
        dataset_name = "_".join([dataset_name, "valid"])
    else:
        dataset_name = "_".join([dataset_name, "test"])

    if binary:
        dataset_name = "_".join([dataset_name, "binary"])

    if multiclass:
        dataset_name = "_".join([dataset_name, "multiclass"])

    if recency:
        dataset_name = "_".join([dataset_name, "recency"])

    results_fname = "_".join(["lgb", strategy, dataset_name, "results.p"])

    results = {}

    results["ndgc"] = ndgc
    results["hr"] = hr

    pickle.dump(results, open(RESULTS_DIR / results_fname, "wb"))


if __name__ == "__main__":

    combinations = [
        # strategy, dataset_name, is_valid, binary, multiclass, recency
        ("leave_one_out", "full", True, True, False, False),
        ("leave_one_out", "full", True, False, False, True),
        ("leave_one_out", "full", False, True, False, False),
        ("leave_one_out", "full", False, False, False, True),
    ]

    for strategy, dataset_name, is_valid, binary, multiclass, recency in combinations:
        hr, ndgc = experiment_metrics(
            strategy, dataset_name, is_valid, binary, multiclass, recency
        )
        save_results(
            hr, ndgc, strategy, dataset_name, is_valid, binary, multiclass, recency
        )

import gc
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# from time import time


pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

PROCESSED_DATA_DIR = Path("data/processed/amazon")
RESULTS_DIR = Path("results")


def load_data(strategy, dataset_name, is_valid, binary, multiclass, recency):

    assert (recency + binary + multiclass) == 1

    if recency:
        dataset_name = "_".join(["recency", dataset_name])

    if is_valid:
        tr_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "train_valid", "le.f"]
        )
        te_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "valid", "le.f"]
        )
    else:
        tr_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "train_test", "le.f"]
        )
        te_fname = "_".join(
            ["gbm", strategy, "w_negative", dataset_name, "test", "le.f"]
        )

    train = pd.read_feather(PROCESSED_DATA_DIR / tr_fname)
    test = pd.read_feather(PROCESSED_DATA_DIR / te_fname)

    if binary:
        train["score"] = train.rating.apply(lambda x: 1 if x > 0 else 0)
        test["score"] = test.rating.apply(lambda x: 1 if x > 0 else 0)
        metric = "binary_logloss"
    elif multiclass:
        train["score"] = train.overall.apply(
            lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3
        )
        test["score"] = test.overall.apply(
            lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3
        )
        metric = "l2"
    elif recency:
        train.rename(columns={"rating": "score"}, inplace=True)
        test.rename(columns={"rating": "score"}, inplace=True)
        metric = "l2"

    try:
        train.drop("rating", axis=1, inplace=True)
        test.drop("rating", axis=1, inplace=True)
    except Exception:
        pass

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    return train, test, metric


def run_lightgbm(
    train, test, metric, strategy, dataset_name, is_valid, binary, multiclass, recency
):

    # test_neg_sample = test[test.score == 0].groupby("user").sample(4)
    # test_pos = test_pos = test[test.score == 1]
    # eval_set = (
    #     pd.concat([test_pos, test_neg_sample]).reset_index(drop=True).sample(frac=1)
    # )

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
    model_fname = "_".join(["lgb", strategy, dataset_name, "model.p"])
    if not is_valid:
        load_model = model_fname.replace("test", "valid")

    cat_cols = [
        "user",
        "item",
        "brand",
        "category",
        "language",
        "verified",
        "dow",
        "dom",
        "top_category",
        "top_brand",
        "top_language",
        "top_topic1",
        "top_topic2",
        "top_umap1",
        "top_umap2",
    ]
    num_cols = [c for c in train.columns if c != "score" and c not in cat_cols]
    target = ["score"]

    lgbtrain = lgb.Dataset(
        data=train[cat_cols + num_cols],
        label=train[target],
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    lgbeval = lgb.Dataset(
        data=test[cat_cols + num_cols],
        label=test[target],
        reference=lgbtrain,
        free_raw_data=False,
    )

    del train, test
    gc.collect()

    if is_valid:
        n_estimators = 1000
    else:
        n_estimators = pickle.load(
            open(PROCESSED_DATA_DIR / load_model, "rb")
        ).best_iteration

    params = {
        "n_estimators": n_estimators,
        "num_leaves": 63,
        "metric": metric,
        "is_unbalance": True if not recency else False,
    }

    lgb_model = lgb.train(
        params,
        lgbtrain,
        valid_sets=[lgbeval],
        valid_names=["eval"],
        early_stopping_rounds=50,  # if is_valid else None,
        verbose_eval=True,
    )

    del lgbtrain
    gc.collect()

    preds = lgb_model.predict(lgbeval.data)
    eval_preds = lgbeval.data[["user", "item"]]
    eval_preds["score"] = lgbeval.label
    eval_preds["preds"] = preds
    eval_preds.reset_index(drop=True, inplace=True)

    del lgbeval
    gc.collect()

    eval_preds.to_feather(PROCESSED_DATA_DIR / preds_fname)
    pickle.dump(lgb_model, open(PROCESSED_DATA_DIR / model_fname, "wb"))


if __name__ == "__main__":
    # strategy = "leave_one_out"
    # dataset_name = "full"
    # is_valid = True
    # binary = True
    # multiclass = False
    # recency = False

    # for the time being going to leave the 5core dataset and leave_n_out out
    # the GBM experiment
    combinations = [
        # strategy, dataset_name, is_valid, binary, multiclass, recency
        ("leave_one_out", "full", True, True, False, False),
        ("leave_one_out", "full", True, False, False, True),
        ("leave_one_out", "full", False, True, False, False),
        ("leave_one_out", "full", False, False, False, True),
    ]

    for strategy, dataset_name, is_valid, binary, multiclass, recency in combinations:

        train, test, metric = load_data(
            strategy, dataset_name, is_valid, binary, multiclass, recency
        )

        run_lightgbm(
            train,
            test,
            metric,
            strategy,
            dataset_name,
            is_valid,
            binary,
            multiclass,
            recency,
        )

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

PROCESSED_DATA_DIR = Path("../data/processed/amazon")
IS_VALID = True
STRATEGY = "leave_one_out"


def test_urm(train_df, data, urm, user_idx, item_idx):
    out = []
    sample_of_user = np.random.choice(train_df.user.unique(), 100)
    df_sample = train_df[train_df.user.isin(sample_of_user)]
    df_sample["user"] = df_sample["user"].map(user_idx).astype(int)
    df_sample["item"] = df_sample["item"].map(item_idx).astype(int)
    user_groups = df_sample.groupby("user")
    for _, gr in user_groups:
        user = gr.user.unique()
        urm_items = np.sort(np.where(urm[user].todense())[1])
        gr_items = np.sort(gr.item)
        out.append(np.all(urm_items == gr_items))
    return np.all(out)


if __name__ == "__main__":

    # datasets = ["full", "5core"]
    datasets = ["full"]

    for dataset in datasets:
        print(
            "INFO: testing that the interactions matrix for the {} dataset was built correctly".format(
                dataset
            )
        )

        if IS_VALID:
            train_df = pd.read_feather(
                PROCESSED_DATA_DIR / "_".join([STRATEGY, "tr", dataset + ".f"])
            )
            input_fname = "_".join([STRATEGY, "w_negative", dataset, "valid.npz"])
            urm_fname = "_".join(["URM", STRATEGY, dataset, "valid.npz"])
            user_idx_fname = "_".join(
                ["user_idx", STRATEGY, "w_negative", dataset, "valid.p"]
            )
            item_idx_fname = "_".join(
                ["item_idx", STRATEGY, "w_negative", dataset, "valid.p"]
            )
        else:
            train_df = pd.read_feather(
                PROCESSED_DATA_DIR / "_".join([STRATEGY, "tr", dataset + ".f"])
            )
            valid_df = pd.read_feather(
                PROCESSED_DATA_DIR / "_".join([STRATEGY, "val", dataset + ".f"])
            )
            train_df = pd.concat([train_df, valid_df])
            input_fname = "_".join([STRATEGY, "w_negative", dataset, "test.npz"])
            urm_fname = "_".join(["URM", STRATEGY, dataset, "test.npz"])
            user_idx_fname = "_".join(
                ["user_idx", STRATEGY, "w_negative", dataset, "test.p"]
            )
            item_idx_fname = "_".join(
                ["item_idx", STRATEGY, "w_negative", dataset, "test.p"]
            )
        urm = load_npz(PROCESSED_DATA_DIR / urm_fname)
        data = np.load(PROCESSED_DATA_DIR / input_fname)
        user_idx = pickle.load(open(PROCESSED_DATA_DIR / user_idx_fname, "rb"))
        item_idx = pickle.load(open(PROCESSED_DATA_DIR / item_idx_fname, "rb"))

        if test_urm(train_df, data, urm, user_idx, item_idx):
            print(
                "INFO: The interactions matrix for the {} dataset was built correctly".format(
                    dataset
                )
            )
        else:
            print("Something went wrong")

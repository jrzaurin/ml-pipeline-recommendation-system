import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import lil_matrix, save_npz

PROCESSED_DATA_DIR = Path("data/processed/amazon")


def built_interaction_mtx(df, dataset, min_reviews_per_user=None):

    if min_reviews_per_user is not None:
        dataset = "_".join([dataset, str(min_reviews_per_user)])
    interactions_mtx_fname = "_".join(["interactions_mtx", dataset + ".npz"])
    users_idx_fname = "_".join(["users_idx", dataset + ".p"])
    items_idx_fname = "_".join(["items_idx", dataset + ".p"])

    if min_reviews_per_user is not None:
        df = df[
            df.reviewerID.isin(
                df.reviewerID.value_counts()[
                    df.reviewerID.value_counts() >= min_reviews_per_user
                ].index.tolist()
            )
        ]

    interactions = df[["reviewerID", "asin"]]
    users = interactions.reviewerID.unique()
    items = interactions.asin.unique()
    users_idx = {k: v for v, k in enumerate(users)}
    items_idx = {k: v for v, k in enumerate(items)}
    pickle.dump(users_idx, open(PROCESSED_DATA_DIR / users_idx_fname, "wb"))
    pickle.dump(items_idx, open(PROCESSED_DATA_DIR / items_idx_fname, "wb"))

    # lil_matrix for speed...
    interactions_mtx = lil_matrix((users.shape[0], items.shape[0]), dtype="float32")

    for j, (_, row) in enumerate(interactions.iterrows()):
        if j % 100000 == 0:
            print(
                "INFO: filled {} out of {} interactions".format(
                    j, interactions.shape[0]
                )
            )
        u = users_idx[row["reviewerID"]]
        i = items_idx[row["asin"]]
        interactions_mtx[u, i] = 1.0

    # ...and csr to save it (save lil format is not implemented)
    interactions_mtx = interactions_mtx.tocsr()

    save_npz(PROCESSED_DATA_DIR / interactions_mtx_fname, interactions_mtx)


if __name__ == '__main__':

    print("INFO: building interactions matrix for the full dataset...")
    train = pd.read_feather(PROCESSED_DATA_DIR / 'train_full.f')
    built_interaction_mtx(train, dataset="full", min_reviews_per_user=3)
    # built_interaction_mtx(train, dataset="full", min_reviews_per_user=5)
    # built_interaction_mtx(train, dataset="full", min_reviews_per_user=7)

    print("INFO: building interactions matrix for the 5 score dataset...")
    train = pd.read_feather(PROCESSED_DATA_DIR / 'train_5core.f')
    built_interaction_mtx(train, dataset="5core", min_reviews_per_user=3)
    # built_interaction_mtx(train, dataset="5core", min_reviews_per_user=5)
    # built_interaction_mtx(train, dataset="5core", min_reviews_per_user=7)

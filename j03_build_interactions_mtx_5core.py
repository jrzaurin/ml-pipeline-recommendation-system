import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import lil_matrix, save_npz

PROCESSED_DATA_DIR = Path("data/processed/amazon")

train = pd.read_feather(PROCESSED_DATA_DIR / "train_5.f")

train = train[
    train.reviewerID.isin(
        train.reviewerID.value_counts()[
            train.reviewerID.value_counts() >= 5
        ].index.tolist()
    )
]

interactions = train[["reviewerID", "asin", "reviewDate", "overall"]]
interactions["overall"] = interactions.overall.apply(
    lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3
)


users = interactions.reviewerID.unique()
items = interactions.asin.unique()

# user and item dictionary of indexes
users_idx = {k: v for v, k in enumerate(users)}
items_idx = {k: v for v, k in enumerate(items)}

# lil_matrix for speed...
interactions_mtx_binary = lil_matrix((users.shape[0], items.shape[0]), dtype="float32")
interactions_mtx_score = lil_matrix((users.shape[0], items.shape[0]), dtype="float32")
for j, (_, row) in enumerate(interactions.iterrows()):
    if j % 100000 == 0:
        print("INFO: filled {} out of {} interactions".format(j, interactions.shape[0]))
    u = users_idx[row["reviewerID"]]
    i = items_idx[row["asin"]]
    score = row["overall"]
    interactions_mtx_binary[u, i] = 1.0
    interactions_mtx_score[u, i] = score

# ...and csr to save it (save lil format is not implemented)
interactions_mtx_binary = interactions_mtx_binary.tocsr()
interactions_mtx_score = interactions_mtx_score.tocsr()

pickle.dump(users_idx, open(PROCESSED_DATA_DIR / "users_idx_5.p", "wb"))
pickle.dump(items_idx, open(PROCESSED_DATA_DIR / "items_idx_5.p", "wb"))
save_npz(PROCESSED_DATA_DIR / "interactions_mtx_binary_5.npz", interactions_mtx_binary)
save_npz(PROCESSED_DATA_DIR / "interactions_mtx_score_5.npz", interactions_mtx_score)

import pickle
from pathlib import Path

from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

PROCESSED_DATA_DIR = Path("data/processed/amazon")


# let's load the activity matrix and dict of indexes
interactions_mtx_score = load_npz(PROCESSED_DATA_DIR / "interactions_mtx_score.npz")

# We built the matrix as user x items, but for knn item based CF we need items x users
interactions_mtx_knn = interactions_mtx_score.T

# users and items indexes
items_idx = pickle.load(open(PROCESSED_DATA_DIR / "items_idx.p", "rb"))
idx_item = {v: k for k, v in items_idx.items()}
users_idx = pickle.load(open(PROCESSED_DATA_DIR / "users_idx.p", "rb"))

# Let's build the KNN model...two lines :)
model = NearestNeighbors()
model.fit(interactions_mtx_knn)

# TODO
# load valid
# keep in valid only those users/items that are in training
# recommend 10, 20, 40, 100
# compute metrics
# compare with most popular

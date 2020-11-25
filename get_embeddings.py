import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gmf import GMF
from sklearn.neighbors import NearestNeighbors

RAW_DATA_DIR = Path("data/raw/amazon")
PROCESSED_DATA_DIR = Path("data/processed/amazon")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")

item_idx = pickle.load(
    open(PROCESSED_DATA_DIR / "item_idx_leave_one_out_w_negative_full_valid.p", "rb")
)

meta = pd.read_pickle(PROCESSED_DATA_DIR / "item_features.p")[["item", "title"]]
keep_cols = ["asin", "title"]
meta = meta[meta.item.isin(item_idx.keys())]
meta["item"] = meta.item.map(item_idx)
item2title = dict(meta.values)

dataset = np.load(PROCESSED_DATA_DIR / "leave_one_out_w_negative_full_valid.npz")
n_users, n_items, n_emb = dataset["n_users"], dataset["n_items"], 8


gmf_model = GMF(n_users, n_items, n_emb)
gmf_model.load_state_dict(torch.load(MODEL_DIR / "gmf_2020-11-20_08:13:23.541131.pt"))
item_embeddings = gmf_model.embeddings_item.weight.data.numpy()

knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(item_embeddings)


def get_movie_titles(input_id, n=20):
    r"""first movie will be the "query" movie and the remaining n-1 the similar
    movies. Similar defined under the functioning of the algorithm, i.e.
    leading to the same prediction"""
    dist, nnidx = knn_model.kneighbors(
        item_embeddings[input_id].reshape(1, -1), n_neighbors=n
    )
    titles = []
    for idx in nnidx[0]:
        try:
            titles.append(item2title[idx])
        except:  # noqa: E722
            continue
    return titles


similar_movies = get_movie_titles(1552)
# ['Rambo 1: First Blood VHS',
#  'Point Break VHS',
#  'Road House VHS',
#  'Crimson Tide VHS',
#  'Easy Rider VHS',
#  'Commando VHS',
#  'Cliffhanger VHS',
#  'Con Air VHS',
#  'Armageddon VHS',
#  'Back To School',
#  'The Rock VHS',
#  'The Last Action Hero VHS',
#  'Terminator, The',
#  'Collateral',
#  'Eraser VHS',
#  'Face/Off VHS',
#  'Beverly Hills Cop VHS',
#  'Trading Places VHS',
#  'Vanishing Point VHS',
#  'Blue Thunder']

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

X = np.random.rand(1000, 10)
model = NearestNeighbors(metric="cosine", n_neighbors=X.shape[0])
model.fit(X)

out = []
for _ in range(10):
    tr_idx = np.random.choice(range(X.shape[0]), 1)
    val_idx = np.setdiff1d(np.arange(X.shape[0]), tr_idx)
    np.random.shuffle(val_idx)
    val_idx = np.asarray([val_idx[0]])
    dist, idx = model.kneighbors(X[tr_idx], n_neighbors=X.shape[0])
    kdist = dist[0][np.where(idx[0] == val_idx)[0]]
    pdist = pairwise_distances(
        X[tr_idx].reshape(1, -1), X[val_idx].reshape(1, -1), metric="cosine"
    )[0]
    out.append(np.isclose(kdist, pdist))
print(np.all(out))

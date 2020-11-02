from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, save_npz

PROCESSED_DATA_DIR = Path("data/processed/amazon")
IS_VALID = True
STRATEGY = "leave_one_out"


def built_interaction_mtx(interactions, output_fname):
    """
    build a binary interaction matrix (or user ratings matrix, URM)
    """

    n_users = interactions.user.nunique()
    n_items = interactions.item.nunique()
    print("INFO: N users: {}. N items: {}.".format(n_users, n_items))

    # lil_matrix for speed...
    interactions_mtx = lil_matrix((n_users, n_items), dtype="float32")

    for j, (_, row) in enumerate(interactions.iterrows()):
        if j % 100000 == 0:
            print(
                "INFO: filled {} out of {} interactions".format(
                    j, interactions.shape[0]
                )
            )
        u = row["user"]
        i = row["item"]
        interactions_mtx[u, i] = 1.0

    # ...and csr to save it (save lil format is not implemented)
    interactions_mtx = interactions_mtx.tocsr()

    save_npz(PROCESSED_DATA_DIR / output_fname, interactions_mtx)


if __name__ == "__main__":

    for dataset in ["full", "5core"]:
        print("INFO: building interactions matrix for the {} dataset.".format(dataset))

        # 1) URM = user ratings matrix 2) for now leave_one_out always implies
        # with w_negative
        if IS_VALID:
            input_fname = "_".join([STRATEGY, "w_negative", dataset, "valid.npz"])
            output_fname = "_".join(["URM", STRATEGY, dataset, "valid.npz"])
        else:
            input_fname = "_".join([STRATEGY, "w_negative", dataset, "test.npz"])
            output_fname = "_".join(["URM", STRATEGY, dataset, "test.npz"])

        train = pd.DataFrame(
            np.load(PROCESSED_DATA_DIR / input_fname)["train"],
            columns=["user", "item", "rating"],
        )

        built_interaction_mtx(train, output_fname)

from pathlib import Path
from time import time

import pandas as pd
import umap
import umap.plot
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
PROCESSED_DATA_DIR = Path("data/processed/amazon")
MAX_VOCAB_SIZE = 10000

if __name__ == "__main__":

    tokenized_descriptions = pd.read_pickle(
        PROCESSED_DATA_DIR / "tokenized_descriptions.p"
    )

    item_desc = []
    for desc in tokenized_descriptions.description.tolist():
        item_desc.append([tok for tok in desc if tok not in STOP_WORDS])

    vectorizer = TfidfVectorizer(
        max_features=MAX_VOCAB_SIZE, preprocessor=lambda x: x, tokenizer=lambda x: x
    )

    X = vectorizer.fit_transform(item_desc)

    for n in [2, 5, 10]:

        print("INFO: running umap with {} components".format(n))

        start = time()

        out_fname = "_".join(["item_desc_tfidf_umap", str(n) + ".f"])
        if n == 2:
            mapper = umap.UMAP(n_neighbors=5, n_components=n, metric="hellinger")
        else:
            mapper = umap.UMAP(n_neighbors=5, n_components=n, metric="cosine")

        X_umap = mapper.fit_transform(X)

        umap_df = pd.DataFrame(X_umap)
        umap_df.columns = ["_".join(["col", str(i)]) for i in range(X_umap.shape[1])]
        umap_df["item"] = tokenized_descriptions["item"]
        umap_df.reset_index(drop=True, inplace=True)
        umap_df.to_feather(PROCESSED_DATA_DIR / out_fname)

        end = time() - start

        print("INFO: umap with {} components took {} min".format(n, round(end / 60, 3)))

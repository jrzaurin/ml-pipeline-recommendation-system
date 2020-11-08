from pathlib import Path
from time import time

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
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

    vectorizer = CountVectorizer(
        max_features=MAX_VOCAB_SIZE, preprocessor=lambda x: x, tokenizer=lambda x: x
    )

    X = vectorizer.fit_transform(item_desc)

    for n in [5, 10, 20]:

        print("INFO: running LDA with {} components".format(n))

        start = time()

        out_fname = "_".join(["item_desc_lda", str(n) + ".f"])

        lda_model = LatentDirichletAllocation(
            n_components=n,
            learning_method="online",
            learning_offset=5.0,
            max_iter=50,
            batch_size=1024,
            evaluate_every=5,
            perp_tol=25.0,
            n_jobs=-1,
            verbose=1,
            random_state=1,
        )

        X_lda = lda_model.fit_transform(X)

        lda_df = pd.DataFrame(X_lda)
        lda_df.columns = ["_".join(["col", str(i)]) for i in range(X_lda.shape[1])]
        lda_df["item"] = tokenized_descriptions["item"]
        lda_df.reset_index(drop=True, inplace=True)
        lda_df.to_feather(PROCESSED_DATA_DIR / out_fname)

        end = time() - start

        print("INFO: LDA with {} components took {} min".format(n, round(end / 60, 3)))

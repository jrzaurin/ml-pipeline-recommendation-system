import pickle
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from gmf import early_stopping, evaluate, train
from ncf_parsers import mlp_parse_args

use_cuda = torch.cuda.is_available()


class MLP(nn.Module):
    """
    Concatenate Embeddings that are then passed through a series of Dense
    layers
    """

    def __init__(self, n_user, n_item, layers, dropouts):
        super(MLP, self).__init__()

        self.layers = layers
        self.n_layers = len(layers)
        self.dropouts = dropouts
        self.n_user = n_user
        self.n_item = n_item

        self.embeddings_user = nn.Embedding(n_user, int(layers[0] / 2))
        self.embeddings_item = nn.Embedding(n_item, int(layers[0] / 2))

        self.mlp = nn.Sequential()
        for i in range(1, self.n_layers):
            self.mlp.add_module("linear%d" % i, nn.Linear(layers[i - 1], layers[i]))
            self.mlp.add_module("leakyrelu%d" % i, nn.LeakyReLU())
            self.mlp.add_module("dropout%d" % i, nn.Dropout(p=dropouts[i - 1]))

        self.out = nn.Linear(in_features=layers[-1], out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

    def forward(self, users, items):

        user_emb = self.embeddings_user(users)
        item_emb = self.embeddings_item(items)
        emb_vector = torch.cat([user_emb, item_emb], dim=1)
        emb_vector = self.mlp(emb_vector)
        preds = torch.sigmoid(self.out(emb_vector))

        return preds


if __name__ == "__main__":  # noqa: C901

    args = mlp_parse_args()

    DATA_DIR = Path(args.datadir)
    MODEL_DIR = Path("models")
    RESULTS_DIR = Path("results")
    dataname = args.dataname
    model_name = "_".join(["mlp", str(datetime.now()).replace(" ", "_")])

    # model params
    layers = eval(args.layers)
    dropouts = eval(args.dropouts)

    # train params
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learner = args.learner
    l2reg = args.l2reg
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    lr_patience = args.lr_patience

    # eval params
    topk = args.topk
    n_neg = args.n_neg
    eval_every = args.eval_every
    early_stop_patience = args.early_stop_patience

    # build or load train and test datasets
    dataset = np.load(DATA_DIR / dataname)
    train_wo_neg = pd.DataFrame(dataset["train"], columns=["user", "item", "rating"])
    test_w_neg = pd.DataFrame(dataset["test"], columns=["user", "item", "rating"])
    # we will treat it as a binary problem: interaction or not
    train_wo_neg["rating"] = train_wo_neg.rating.apply(lambda x: 1 if x > 0 else x)
    test_w_neg["rating"] = test_w_neg.rating.apply(lambda x: 1 if x > 0 else x)
    n_users, n_items = dataset["n_users"], dataset["n_items"]

    # test loader for validation
    eval_loader = DataLoader(dataset=test_w_neg.values, batch_size=1000, shuffle=False)

    # model definition
    model = MLP(n_users, n_items, layers, dropouts)
    if use_cuda:
        model = model.cuda()

    if learner.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2reg)
    else:
        optimizer = torch.optim.SGD(  # type: ignore[assignment]
            model.parameters(), lr=lr, weight_decay=l2reg, momentum=0.9, nesterov=True
        )

    if lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=lr_patience,
            factor=0.5,
            threshold=0.001,
            threshold_mode="rel",
        )

    best_score = -np.inf
    stop_step = 0
    update_count = 0
    stop = False
    for epoch in range(n_epochs):
        t1 = time()
        train(
            model,
            optimizer,
            epoch,
            batch_size,
            train_wo_neg,
            test_w_neg,
            n_items,
            n_neg,
        )
        t2 = time()
        if epoch % eval_every == (eval_every - 1):
            hr, ndcg = evaluate(model, eval_loader, topk)

            early_stop_score = ndcg
            best_score, stop_step, stop = early_stopping(
                early_stop_score,
                best_score,
                stop_step,
                early_stop_patience,
            )
            if lr_scheduler:
                scheduler.step(early_stop_score)
            print("=" * 80)
            print(
                "HR = {:.4f}, NDCG = {:.4f}, validated in {:.2f}s".format(
                    hr, ndcg, time() - t2
                )
            )
            print("=" * 80)
        if stop:
            break
        if (stop_step == 0) & (args.save_results):
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_DIR / (model_name + ".pt"))

    if args.save_results:
        # Save results
        results_d = {}
        results_d["args"] = args.__dict__
        results_d["best_epoch"] = best_epoch
        results_d["ndcg"] = ndcg
        results_d["hr"] = hr
        pickle.dump(results_d, open(str(RESULTS_DIR / (model_name + ".p")), "wb"))

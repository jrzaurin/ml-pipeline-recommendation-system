import os
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

from gmf import GMF, early_stopping, evaluate, train
from mlp import MLP
from ncf_parsers import ncf_parse_args

use_cuda = torch.cuda.is_available()


class NeuMF(nn.Module):
    def __init__(self, n_user, n_item, n_emb, layers, dropouts):
        super(NeuMF, self).__init__()

        self.layers = layers
        self.n_layers = len(layers)
        self.dropouts = dropouts
        self.n_user = n_user
        self.n_item = n_item

        self.mf_embeddings_user = nn.Embedding(n_user, n_emb)
        self.mf_embeddings_item = nn.Embedding(n_item, n_emb)

        self.mlp_embeddings_user = nn.Embedding(n_user, layers[0] // 2)
        self.mlp_embeddings_item = nn.Embedding(n_item, layers[0] // 2)
        self.mlp = nn.Sequential()
        for i in range(1, self.n_layers):
            self.mlp.add_module("linear%d" % i, nn.Linear(layers[i - 1], layers[i]))
            self.mlp.add_module("leakyrelu%d" % i, nn.LeakyReLU())
            self.mlp.add_module("dropout%d" % i, nn.Dropout(p=dropouts[i - 1]))

        self.out = nn.Linear(in_features=n_emb + layers[-1], out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

    def forward(self, users, items):

        mf_user_emb = self.mf_embeddings_user(users)
        mf_item_emb = self.mf_embeddings_item(items)

        mlp_user_emb = self.mlp_embeddings_user(users)
        mlp_item_emb = self.mlp_embeddings_item(items)

        mf_emb_vector = mf_user_emb * mf_item_emb
        mlp_emb_vector = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)
        mlp_emb_vector = self.mlp(mlp_emb_vector)

        emb_vector = torch.cat([mf_emb_vector, mlp_emb_vector], dim=1)
        preds = torch.sigmoid(self.out(emb_vector))

        return preds


def load_pretrain_model(model, gmf_model, mlp_model):

    # MF embeddings
    model.mf_embeddings_item.weight = gmf_model.embeddings_item.weight
    model.mf_embeddings_user.weight = gmf_model.embeddings_user.weight

    # MLP embeddings
    model.mlp_embeddings_item.weight = mlp_model.embeddings_item.weight
    model.mlp_embeddings_user.weight = mlp_model.embeddings_user.weight

    # MLP layers
    model_dict = model.state_dict()
    mlp_layers_dict = mlp_model.state_dict()
    mlp_layers_dict = {k: v for k, v in mlp_layers_dict.items() if "linear" in k}
    model_dict.update(mlp_layers_dict)
    model.load_state_dict(model_dict)

    # Prediction weights
    mf_prediction_weight, mf_prediction_bias = gmf_model.out.weight, gmf_model.out.bias
    mlp_prediction_weight, mlp_prediction_bias = (
        mlp_model.out.weight,
        mlp_model.out.bias,
    )

    new_weight = torch.cat([mf_prediction_weight, mlp_prediction_weight], dim=1)
    new_bias = mf_prediction_bias + mlp_prediction_bias
    model.out.weight = torch.nn.Parameter(0.5 * new_weight)
    model.out.bias = torch.nn.Parameter(0.5 * new_bias)

    return model


if __name__ == "__main__":  # noqa: C901

    args = ncf_parse_args()

    DATA_DIR = Path(args.datadir)
    MODEL_DIR = Path("models")
    RESULTS_DIR = Path("results")
    dataname = args.dataname
    model_name = "_".join(["ncf", str(datetime.now()).replace(" ", "_")])

    # gmf
    n_emb = args.n_emb
    # mlp
    layers = eval(args.layers)
    dropouts = eval(args.dropouts)
    # train/eval parameter
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    learner = args.learner
    l2reg = args.l2reg
    lr_scheduler = args.lr_scheduler
    lr_patience = args.lr_patience
    eval_every = args.eval_every
    n_neg = args.n_neg
    topk = args.topk
    early_stop_patience = args.early_stop_patience
    # Pretrained model names
    freeze = args.freeze
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

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

    model = NeuMF(n_users, n_items, n_emb, layers, dropouts)
    if os.path.isfile(mf_pretrain) and os.path.isfile(mlp_pretrain):
        gmf_model = GMF(n_users, n_items, n_emb)
        gmf_model.load_state_dict(torch.load(mf_pretrain))
        mlp_model = MLP(n_users, n_items, layers, dropouts)
        mlp_model.load_state_dict(torch.load(mlp_pretrain))
        model = load_pretrain_model(model, gmf_model, mlp_model)
        print(
            "Load pretrained GMF {} and MLP {} models done. ".format(
                mf_pretrain, mlp_pretrain
            )
        )

    # model definition
    if use_cuda:
        model = model.cuda()

    if freeze:
        for name, layer in model.named_parameters():
            if not ("out" in name):
                layer.requires_grad = False

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
            factor=0.1,
            threshold=0.001,
            threshold_mode="abs",
            verbose=True,
        )

    best_score = -np.inf
    stop_step = 0
    stop = False
    for epoch in range(n_epochs):
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
        if epoch % eval_every == (eval_every - 1):
            start = time()
            hr, ndcg = evaluate(model, eval_loader, topk)
            print("=" * 80)
            print(
                "HR = {:.4f}, NDCG = {:.4f}, validated in {:.2f}s".format(
                    hr, ndcg, time() - start
                )
            )
            print("=" * 80)
            early_stop_score = ndcg
            best_score, stop_step, stop = early_stopping(
                early_stop_score,
                best_score,
                stop_step,
                early_stop_patience,
            )
            if lr_scheduler:
                scheduler.step(early_stop_score)
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

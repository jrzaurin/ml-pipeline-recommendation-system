import heapq
import math
import os
import pickle
from datetime import datetime
# from functools import partial
# from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from gmf_parser import parse_args

# from metrics import hit_ratio, ndcg_binary

use_cuda = torch.cuda.is_available()


class GMF(nn.Module):
    def __init__(self, n_user, n_item, n_emb=8):
        super(GMF, self).__init__()

        self.n_emb = n_emb
        self.n_user = n_user
        self.n_item = n_item

        self.embeddings_user = nn.Embedding(n_user, n_emb)
        self.embeddings_item = nn.Embedding(n_item, n_emb)
        self.out = nn.Linear(in_features=n_emb, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, users, items):

        user_emb = self.embeddings_user(users)
        item_emb = self.embeddings_item(items)
        prod = user_emb * item_emb
        preds = torch.sigmoid(self.out(prod))

        return preds


def hit_ratio_ncf(ranklist, gtitem):
    if gtitem in ranklist:
        return 1
    return 0


def ndcg_binary_ncf(ranklist, gtitem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtitem:
            return math.log(2) / math.log(i + 2)
    return 0


def get_metrics_ncf(items, preds, topk):

    gtitem = items[0]

    # the following 3 lines of code ensure that the fact that the 1st item is
    # gtitem does not affect the final rank
    randidx = np.arange(100)
    np.random.shuffle(randidx)
    items, preds = items[randidx], preds[randidx]

    map_item_score = dict(zip(items, preds))
    ranklist = heapq.nlargest(topk, map_item_score, key=map_item_score.get)
    hr = hit_ratio_ncf(ranklist, gtitem)
    ndcg = ndcg_binary_ncf(ranklist, gtitem)
    return hr, ndcg


# def get_metrics(group, k):
#     df = group[1]
#     true = df[df.rating != 0]["item"].values
#     rec = df.sort_values("preds", ascending=False)["item"].values[:k]
#     return (hit_ratio(rec, true, k), ndcg_binary(rec, true, k))


def _sample_train_neg_instances(
    train_pos, train_wo_neg_lookup, test_w_neg_lookup, n_items, n_neg
):
    user, item, labels = [], [], []
    for u, i, r in tqdm(train_pos, desc="Sample Train Negatives"):
        # we need to make sure they are not in the negative examples used for
        # testing
        try:
            user_test_neg = test_w_neg_lookup[u]
        except KeyError:
            user_test_neg = [-666]

        for _ in range(n_neg):
            j = np.random.randint(n_items)
            while j in train_wo_neg_lookup[u] or j in user_test_neg:
                j = np.random.randint(n_items)
            user.append(u)
            item.append(j)
            labels.append(0)

    train_w_negative = np.vstack([user, item, labels]).T

    return train_w_negative.astype(np.int64)


def get_train_instances(train_wo_neg, test_w_neg, n_items, n_neg=4):

    train_wo_neg_lookup = train_wo_neg.groupby("user")["item"].apply(list)
    test_w_neg_lookup = test_w_neg.groupby("user")["item"].apply(list)
    train_neg = _sample_train_neg_instances(
        train_wo_neg.values,
        train_wo_neg_lookup,
        test_w_neg_lookup,
        n_items,
        n_neg=n_neg,
    )
    train_w_neg = (
        pd.DataFrame(
            np.vstack([train_wo_neg, train_neg]),
            columns=["user", "item", "rating"],
        )
        .sort_values(["user", "item"])
        .drop_duplicates(["user", "item"])
        .reset_index(drop=True)
    )

    return train_w_neg.values


def train(
    model,
    optimizer,
    epoch,
    batch_size,
    train_wo_neg,
    test_w_neg,
    n_items,
    n_neg,
):
    model.train()
    train_w_neg = get_train_instances(train_wo_neg, test_w_neg, n_items, n_neg)
    train_loader = DataLoader(
        dataset=train_w_neg, batch_size=batch_size, num_workers=4, shuffle=True
    )
    train_steps = (len(train_loader.dataset) // train_loader.batch_size) + 1
    running_loss = 0
    with trange(train_steps) as t:
        for batch_idx, data in zip(t, train_loader):
            t.set_description("epoch: {}".format(epoch + 1))
            users, items, labels = data[:, 0], data[:, 1], data[:, 2].float()
            if use_cuda:
                users, items, labels = users.cuda(), items.cuda(), labels.cuda()
            optimizer.zero_grad()
            preds = model(users, items)
            loss = F.binary_cross_entropy(preds.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            t.set_postfix(loss=avg_loss)
    return avg_loss


# def evaluate(model, eval_loader):
#     model.eval()
#     eval_preds = []
#     with torch.no_grad():
#         for data in tqdm(eval_loader, desc="Valid"):
#             users, items, labels = data[:, 0], data[:, 1], data[:, 2].float()
#             if use_cuda:
#                 users, items, labels = users.cuda(), items.cuda(), labels.cuda()
#             preds = model(users, items)
#             preds_cpu = preds.squeeze(1).detach().cpu().numpy()
#             eval_preds += [preds_cpu]
#     return np.hstack(eval_preds)


def evaluate_ncf(model, eval_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for data in tqdm(eval_loader, desc="Valid"):
            users = data[:, 0]
            items = data[:, 1]
            labels = data[:, 2].float()
            if use_cuda:
                users, items, labels = users.cuda(), items.cuda(), labels.cuda()
            preds = model(users, items)
            preds_cpu = preds.squeeze(1).detach().cpu().numpy()

            split_chuncks = preds_cpu.shape[0] // 100
            items_cpu = items.cpu().numpy()
            item_chunks = np.split(items_cpu, split_chuncks)
            pred_chunks = np.split(preds_cpu, split_chuncks)
            scores += [
                get_metrics_ncf(it, pr, topk)
                for it, pr in zip(item_chunks, pred_chunks)
            ]

    hr = [s[0] for s in scores]
    ndcg = [s[1] for s in scores]
    return (np.array(hr).mean(), np.array(ndcg).mean())


def early_stopping(curr_value, best_value, stop_step, patience):
    if curr_value >= best_value:
        stop_step, best_value = 0, curr_value
    else:
        stop_step += 1
    if stop_step >= patience:
        print(
            "Early stopping triggered. patience: {} log:{}".format(patience, best_value)
        )
        stop = True
    else:
        stop = False
    return best_value, stop_step, stop


if __name__ == "__main__":  # noqa: C901

    args = parse_args()

    DATA_DIR = Path(args.datadir)
    MODEL_DIR = Path("models")
    RESULTS_DIR = Path("results")
    dataname = args.dataname
    model_name = "_".join(["gmf", str(datetime.now()).replace(" ", "_")])

    # model params
    n_emb = args.n_emb

    # train params
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learner = args.learner
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    lr_patience = args.lr_patience

    # eval params
    topk = args.topk
    n_neg = args.n_neg
    eval_every = args.eval_every
    early_stop_patience = args.early_stop_patience

    # save
    save_results = args.save_results

    for d in [RESULTS_DIR, MODEL_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

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
    model = GMF(n_users, n_items, n_emb=n_emb)
    if use_cuda:
        model = model.cuda()

    if learner.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True
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
            # preds = evaluate(model, eval_loader)
            # test_w_neg["preds"] = preds
            # user_groups = test_w_neg.groupby("user")

            # with Pool(cpu_count()) as p:
            #     res = p.map(partial(get_metrics, k=10), [g for g in user_groups])

            # hr = np.mean([el[0] for el in res])
            # ndcg = np.mean([el[1] for el in res])

            hr, ndcg = evaluate_ncf(model, eval_loader)

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
        if (stop_step == 0) & (save_results):
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_DIR / (model_name + ".pt"))

    if save_results:
        # Save results
        results_d = {}
        results_d["args"] = args.__dict__
        results_d["best_epoch"] = best_epoch
        results_d["ndcg"] = ndcg
        results_d["hr"] = hr
        pickle.dump(results_d, open(str(RESULTS_DIR / (model_name + ".p")), "wb"))

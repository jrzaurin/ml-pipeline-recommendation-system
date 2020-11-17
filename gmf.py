import argparse
import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from metrics import hit_ratio, ndcg_binary


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir", type=str, default="data/processed/amazon", help="data directory."
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        default="models",
        help="models directory",
    )
    parser.add_argument(
        "--resultsdir",
        type=str,
        default="results",
        help="results directory",
    )
    parser.add_argument(
        "--dataname",
        type=str,
        default="leave_one_out_w_negative_full_valid.npz",
        help="npz file with dataset",
    )
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size.")
    parser.add_argument("--n_emb", type=int, default=8, help="embedding size.")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="if lr_scheduler this will be max_lr"
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="adamw",
        help="Specify an optimizer: adamw or sgd",
    )
    # parser.add_argument(
    #     "--lr_scheduler",
    #     action="store_true",
    #     help="boolean to set the use of ReduceLROnPlateau during training",
    # )
    parser.add_argument(
        "--validate_every", type=int, default=1, help="validate every n epochs"
    )
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument(
        "--n_neg",
        type=int,
        default=4,
        help="number of negative instances to consider per positive instance",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="number of items to retrieve for recommendation",
    )

    return parser.parse_args()


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


def get_metrics(group, k):
    df = group[1]
    true = df[df.score != 0]["item"].values
    rec = df.sort_values("preds", ascending=False)["item"].values[:k]
    return (hit_ratio(rec, true, k), ndcg_binary(rec, true, k))


def sample_train_neg_instances(
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
    train_neg = sample_train_neg_instances(
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


def checkpoint(model, modelpath):
    torch.save(model.state_dict(), modelpath)


def train(
    model,
    optimizer,
    epoch,
    batch_size,
    use_cuda,
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
            users = data[:, 0]
            items = data[:, 1]
            labels = data[:, 2].float()
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


def evaluate(model, test_loader, use_cuda, topk):
    model.eval()
    eval_preds = []
    with torch.no_grad():
        for data in test_loader:
            users = data[:, 0]
            items = data[:, 1]
            labels = data[:, 2].float()
            if use_cuda:
                users, items, labels = users.cuda(), items.cuda(), labels.cuda()
            preds = model(users, items)
            preds_cpu = preds.squeeze(1).detach().cpu().numpy()
            eval_preds += [preds_cpu]
    return np.vstack(eval_preds)


if __name__ == "__main__":  # noqa: C901

    datadir = "data/processed/amazon"
    modeldir = "models"
    resultsdir = "results"
    dataname = "leave_one_out_w_negative_full_valid.npz"
    n_emb = 8
    lr = 0.01
    batch_size = 1024
    epochs = 1
    learner = "adamw"
    topk = 4
    n_neg = 4
    topk = 10
    validate_every = 1
    save_model = False

    # args = parse_args()

    # # dirs and filenames
    # datadir = args.datadir
    # modeldir = args.modeldir
    # resultsdir = args.resultsdir
    # dataname = args.dataname

    # # model params
    # n_emb = args.n_emb

    # # train params
    # batch_size = args.batch_size
    # epochs = args.epochs
    # learner = args.learner
    # # lr = args.lr
    # # lr_scheduler = args.lr_scheduler
    # # lrs = "wlrs" if lr_scheduler else "wolrs"

    # # eval params
    # topk = args.topk
    # n_neg = args.n_neg
    # validate_every = args.validate_every

    # # save
    # save_model = args.save_model

    modelfname = (
        "GMF"
        + "_".join(["_bs", str(batch_size)])
        # + "_".join(["_lr", str(lr).replace(".", "")])
        + "_".join(["_n_emb", str(n_emb)])
        + "_".join(["_lrnr", learner])
        # + "_".join(["_lrs", lrs])
        + ".pt"
    )
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, "results_df.p")

    # build or load train and test datasets
    dataset = np.load(os.path.join(datadir, dataname))
    train_wo_neg = pd.DataFrame(dataset["train"], columns=["user", "item", "rating"])
    test_w_neg = pd.DataFrame(dataset["test"], columns=["user", "item", "rating"])
    # we will treat it as a binary problem: interaction or not
    train_wo_neg["rating"] = train_wo_neg.rating.apply(lambda x: 1 if x > 0 else x)
    test_w_neg["rating"] = test_w_neg.rating.apply(lambda x: 1 if x > 0 else x)
    n_users, n_items = dataset["n_users"], dataset["n_items"]

    # test loader for validation
    test_loader = DataLoader(dataset=test_w_neg.values, batch_size=1000, shuffle=False)

    # model definition
    model = GMF(n_users, n_items, n_emb=n_emb)

    # set the optimizer and loss
    if learner.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True
        )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # --> HERE: early stop and save results and model
    best_hr, best_ndcgm, best_iter = 0, 0, 0
    for epoch in range(epochs):
        t1 = time()
        train(
            model,
            optimizer,
            epoch,
            batch_size,
            use_cuda,
            train_wo_neg,
            test_w_neg,
            n_items,
            n_neg,
        )
        t2 = time()
        if epoch % validate_every == 0:
            res = evaluate(model, test_loader, use_cuda, topk)
            hr = np.mean([el[0] for el in res])
            ndcg = np.mean([el[1] for el in res])
            print(
                "Epoch: {} {:.2f}s,  HR = {:.4f}, NDCG = {:.4f}, validated in {:.2f}s".format(
                    epoch, t2 - t1, hr, ndcg, time() - t2
                )
            )
            if hr > best_hr:
                best_hr, best_ndcg, best_iter, train_time = (
                    hr,
                    ndcg,
                    epoch,
                    t2 - t1,
                )
                if save_model:
                    checkpoint(model, modelpath)

    print(
        "End. Best Iteration {}: HR = {:.4f}, NDCG = {:.4f}. ".format(
            best_iter, best_hr, best_ndcg
        )
    )
    if save_model:
        print("The best GMF model is saved to {}".format(modelpath))

    # if save_model:
    #     cols = [
    #         "modelname",
    #         "iter_loss",
    #         "best_hr",
    #         "best_ndcg",
    #         "best_iter",
    #         "train_time",
    #     ]
    #     vals = [modelfname, iter_loss, best_hr, best_ndcg, best_iter, train_time]
    #     if not os.path.isfile(resultsdfpath):
    #         results_df = pd.DataFrame(columns=cols)
    #         experiment_df = pd.DataFrame(data=[vals], columns=cols)
    #         results_df = results_df.append(experiment_df, ignore_index=True)
    #         results_df.to_pickle(resultsdfpath)
    #     else:
    #         results_df = pd.read_pickle(resultsdfpath)
    #         experiment_df = pd.DataFrame(data=[vals], columns=cols)
    #         results_df = results_df.append(experiment_df, ignore_index=True)
    #         results_df.to_pickle(resultsdfpath)

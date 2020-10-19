import bottleneck as bn
import numpy as np


def ndcg_binary_at_k_batch(X_pred, heldout_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance

    This implementation is designed for binary relevance and to take matrices

    X_pred: np.array
        array with shape (batch size, N_items)
    heldout_batch: sparse mtx
        array with shape (batch size, N_items)
    """

    # all this is an efficient np.argsort(-X_pred)[:, :k] for when we have
    # lots of items
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    # build the discount template
    tp = 1.0 / np.log2(np.arange(2, k + 2))

    DCG = (
        heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp
    ).sum(axis=1)
    IDCG = np.array([(tp[: min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return DCG[IDCG > 0.0] / IDCG[IDCG > 0.0]


def ndcg_score_at_k_batch(X_pred, heldout_batch, k=100):
    # TO DO: FINISH
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    # build the discount template
    tp = 1.0 / np.log2(np.arange(2, k + 2))

    DCG = (
        heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp
    ).sum(axis=1)

    return DCG


def recall_binary_at_k_batch(X_pred, heldout_batch, k=100):
    """
    Recall@k

    This implementation is designed to take matrices

    X_pred: np.array
        array with shape (batch size, N_items)
    heldout_batch: sparse mtx
        array with shape (batch size, N_items)
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    allp = np.minimum(k, X_true_binary.sum(axis=1))
    recall = tp[allp > 0.0] / allp[allp > 0.0]
    return recall


def dcg_scores_at_k(y_true, y_pred, method=1, k=10):
    """
    Computes the discounted cumulative gain @ k.

    This function is designed to take dictionarires with the actual and
    predicted interactions. Is also designed to take the score that the user
    gave to the item (as opposed as simply consider it a binary problem)

    actual : dict
        A dict of elements: {'item': score} for the actual interactions
    predicted : dict
        A dict of predicted elements {'item': score}
    k : int, optional
        The maximum number of predicted elements
    """
    # First rank the recommendations based on predicted scores
    ranked_rec = sorted(
        [(k, v) for k, v in y_pred.items()], key=lambda x: x[1], reverse=True
    )
    # Select those that the user did interact with and their rank
    ranked_rec = [(i, k) for i, (k, v) in enumerate(ranked_rec) if k in y_true.keys()]
    # then extract the rank and the relevance (real score) and keep min(len(y_true), k)
    ranked_scores = [(k, y_true[v]) for k, v in dict(ranked_rec).items()][
        : min(len(y_true), k)
    ]

    if not ranked_scores:
        return 0.0
    else:
        rank = np.asarray([s[0] for s in ranked_scores])
        score = np.asarray([s[1] for s in ranked_scores])
        return np.sum((2 ** score - 1) / np.log2(rank + 1))


def ndcg_scores_at_k(y_true, y_pred, method=1, k=10):
    return dcg_scores_at_k(y_true, y_pred) / dcg_scores_at_k(y_true, y_true)


def mean_ndcg_scores_at_k(actual, predicted, method=1, k=10):
    return np.mean([ndcg_scores_at_k(t, p, k) for t, p in zip(actual, predicted)])

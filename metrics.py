import bottleneck as bn
import numpy as np


def ndcg_binary_at_k_batch(X_pred, heldout_batch, k):
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ranking = _efficient_sort(X_pred, 2)
    dcg = (
        heldout_batch[np.arange(ranking.shape[0])[:, np.newaxis], ranking].toarray()
        * discount
    ).sum(axis=1)
    idcg = np.array(
        [(discount[: min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)]
    )
    return dcg[idcg > 0.0] / idcg[idcg > 0.0]


def recall_binary_at_k_batch(X_pred, heldout_batch, k):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    allp = np.minimum(k, X_true_binary.sum(axis=1))
    recall = tp[allp > 0.0] / allp[allp > 0.0]
    return recall


def ndcg_score_at_k_batch(y_true, y_score, k):
    dcg = _dcg_sample_scores(y_true, y_score, k, ignore_ties=True)
    idcg = _dcg_sample_scores(y_true, y_true, k, ignore_ties=True)
    return dcg[idcg > 0.0] / idcg[idcg > 0.0]


def hit_ratio_at_k_batch(X_pred, heldout_batch, k):
    ranking = _efficient_sort(X_pred, k)
    hr = heldout_batch[np.arange(ranking.shape[0])[:, np.newaxis], ranking].getnnz(
        axis=1
    )
    return hr


def ndgc_binary(rec, true, k):
    dcg = _dcg_binary(rec, true, k)
    idcg = _dcg_binary(true, true, k)
    return dcg / idcg


def ndgc_score(rec, true, true_relevance, k):
    dcg = _dcg_binary(rec, true, true_relevance, k)
    idcg = _dcg_binary(true, true, true_relevance, k)
    return dcg / idcg


def hit_ratio(rec, true, k):
    return np.intersect1d(rec[:k], true).shape[0]


def recall_binary(rec, true, k):
    tp = np.intersect1d(rec[:k], true).shape[0]
    allp = true.shape[0]
    return tp / allp


def _dcg_binary(rec, true, k):
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    _, rank, _ = np.intersect1d(rec[:k], true, return_indices=True)
    dcg = discount[rank].sum()
    return dcg


def _dcg_score(rec, true, true_score, k):
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    rec_item, rec_rank, true_loc = np.intersect1d(rec[:k], true, return_indices=True)
    true_relevance = true_score[true_loc]
    discount = discount[rec_rank]
    dcg = true_relevance * discount
    return dcg


def _dcg_sample_scores(y_true, y_score, k, ignore_ties=True):
    if ignore_ties:
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        # ranking = _efficient_sort(y_score, k)
        ranking = np.argsort(y_score)[:, ::-1][:, :k]
        ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
        cumulative_gains = discount.dot(ranked.T)
    else:
        discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(2))
        discount[k:] = 0
        discount_cumsum = np.cumsum(discount)
        cumulative_gains = [
            _tie_averaged_dcg(y_t, y_s, discount_cumsum)
            for y_t, y_s in zip(y_true, y_score)
        ]
        cumulative_gains = np.asarray(cumulative_gains)
    return cumulative_gains


def _tie_averaged_dcg(y_true, y_score, discount_cumsum):
    _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
    ranked = np.zeros(len(counts))
    np.add.at(ranked, inv, y_true)
    ranked /= counts
    groups = np.cumsum(counts) - 1
    discount_sums = np.empty(len(counts))
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    return (ranked * discount_sums).sum()


def _efficient_sort(arr, k):
    batch_size = arr.shape[0]
    idx_topk_part = bn.argpartition(-arr, k, axis=1)
    topk_part = arr[np.arange(batch_size)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_size)[:, np.newaxis], idx_part]
    return idx_topk

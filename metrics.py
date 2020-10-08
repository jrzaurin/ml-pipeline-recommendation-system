import json
from datetime import timedelta
from datetime import datetime
import numpy as np

import luigi
import dask.dataframe as dd

from utils import Mario
from x04_train_test_split import DEV_SET_REVIEWS_JOB
from baselines import MostPopularReco


def dcg(recs, relevance):
    """calculates Discounted Cumulative Gain

    takes two dataframes - of recommendations with columns
    - item
    - reviewerID
    - rank

    and another of item relevance with columns:
    - item
    - reviewerID
    - relevance

    Returns dataframe with columns
    - reviewerID
    - DCG
    """
    recs = recs[['item', 'reviewerID', 'rank']]
    recs['discount_factor'] = recs['rank'].map(lambda x: np.log2(x + 1))
    hits = relevance.merge(recs, on=['item', 'reviewerID'])
    hits['DCG'] = hits.relevance / hits.discount_factor
    cumulative = hits.groupby('reviewerID').agg({'DCG': 'sum'}).reset_index()
    all_reviewers = relevance[['reviewerID']].drop_duplicates()
    result = cumulative.merge(
        all_reviewers,
        on='reviewerID',
        how='right').fillna(0)
    return result


def ndcg(recs, relevance, k):
    """calculates Normalized Discounted Cumulative Gain @k

    takes two dataframes - of recommendations with columns
    - item
    - reviewerID
    - rank

    and another of item relevance with columns:
    - item
    - reviewerID
    - relevance
    - rank

    Returns dataframe with columns
    - reviewerID
    - NDCG
    """
    dcg_df = dcg(recs[recs['rank'] <= k], relevance)
    norm = dcg(relevance[relevance['rank'] <= k], relevance) \
        .rename(columns={'DCG': 'NORM'})
    result = dcg_df.merge(norm, on='reviewerID')
    result['NDCG'] = result.DCG / result.NORM
    return result[['reviewerID', 'NDCG', 'DCG']]


class ItemRelevance(Mario, luigi.Task):
    """
    Calculates a measure of relevance of item for a user
    based on the dev set.

    Relevance = 1 if user rated item 1, 2, 3
    Relevance = 2 if user rated item 4, 5
    Relevance = 0 for if use has not rated the item

    If a user has rated the item multiple times, max rating is used.
    0 relevance user-item pairs are not saved.

    item: string
    reviewerID: string
    overall: double
    relevance: int64
    rank: int64
    """

    def output_dir(self):
        return 'dev_metrics/item_relevance'

    def requires(self):
        return DEV_SET_REVIEWS_JOB

    def _run(self):
        user_items = (
            self.requires()
            .load_parquet()
            .groupby(['item', 'reviewerID'])
            .agg({'overall': 'max'})
            .reset_index()
        )
        user_items['relevance'] = user_items.overall.map({
            1: 1,
            2: 1,
            3: 1,
            4: 2,
            5: 2
        })

        # have to break out of dask here because there is no rank function in
        # dask
        user_items_pd = user_items.compute()
        user_items_pd['rank'] = (
            user_items_pd
            .groupby('reviewerID')
            .relevance
            .rank(method='first')
            .astype('int')
        )
        self.save_parquet(dd.from_pandas(user_items_pd, npartitions=20))


class EvaluateReco(Mario, luigi.Task):
    def output_dir(self):
        return 'dev_metrics/eval'

    def requires(self):
        return [ItemRelevance(), MostPopularReco(days=30, k=10)]

    def _run(self):
        relevance_job, reco_job = self.requires()
        relevance = relevance_job.load_parquet()
        recs = reco_job.load_parquet()

        metrics = ndcg(recs, relevance, 10)
        print(metrics.mean().compute())

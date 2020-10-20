import json
from datetime import timedelta
from datetime import datetime
import numpy as np
import pandas as pd

import luigi
import dask.dataframe as dd

from utils import Mario
from x04_train_test_split import FilteredDevSet
from baselines import MostPopularReco, RandomReco, MostPopularInCatReco
from biggraph import PBGReco


def get_reco_job(reco_name, k):
    days = 31
    return {
        'Most Popular': MostPopularReco(days=days, k=k),
        'Most Popular in Cat': MostPopularInCatReco(days=days, k=k),
        'Random': RandomReco(days=days, k=k),
        'PBG V01': PBGReco(
            item_days=days, k=k,
            epochs=2,
            dim=100,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=1000,
            days=2,
            min_user_rev=2
        )
    }[reco_name]


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
    recs = recs[['item', 'reviewerID', 'rank']].reset_index(drop=True)
    recs['discount_factor'] = recs['rank'].map(lambda x: np.log2(x + 1))
    hits = relevance.merge(recs, on=['item', 'reviewerID'])
    hits['DCG'] = hits.relevance / hits.discount_factor
    hits['hits'] = 1
    cumulative = hits.groupby('reviewerID').agg(
        {'DCG': 'sum', 'hits': 'sum'}).reset_index()
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
    - hits
    - DCG
    - NDCG
    """
    dcg_df = dcg(recs[recs['rank'] <= k], relevance)
    norm = dcg(relevance[relevance['rank'] <= k], relevance) \
        .rename(columns={'DCG': 'NORM'})[['reviewerID', 'NORM']]
    result = dcg_df.merge(norm, on='reviewerID')
    result['NDCG'] = result.DCG / result.NORM
    return result[['reviewerID', 'NDCG', 'DCG', 'hits']]


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
        return FilteredDevSet()

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
    """
    NDCG: double
    DCG: double
    hits: double
    Reco: string
    k: int64
    """
    reco_name = luigi.Parameter()
    k = luigi.IntParameter()

    def output_dir(self):
        return 'dev_metrics/eval/%s/k=%s' % (self.reco_name, self.k)

    def requires(self):
        return [
            ItemRelevance(),
            get_reco_job(self.reco_name, self.k)
        ]

    def _run(self):
        relevance_job, reco_job = self.requires()
        relevance = relevance_job.load_parquet().compute()
        recs = reco_job.load_parquet().compute()

        # sanity checks
        all_users = set(relevance.reviewerID)
        assert set(recs.reviewerID) == all_users
        assert set(recs[recs['rank'] == 1].reviewerID) == all_users
        assert set(recs[recs['rank'] == self.k].reviewerID) == all_users

        metrics = ndcg(recs, relevance, self.k)
        results = dict(metrics.mean())
        results['Reco'] = self.reco_name
        results['k'] = self.k
        results_pd = pd.DataFrame([results])
        print(results_pd)
        results_dd = dd.from_pandas(results_pd, npartitions=1)
        self.save_parquet(results_dd)


class EvalEverything(Mario, luigi.Task):
    """
    NDCG: double
    DCG: double
    hits: double
    Reco: string
    k: int64
    """

    def output_dir(self):
        return 'dev_metrics/all_experiments'

    def requires(self):
        reco_names = [
            'Most Popular',
            'Random',
            'Most Popular in Cat',
            'PBG V01'
        ]
        for k in [10, 20, 40]:
            for name in reco_names:
                yield EvaluateReco(reco_name=name, k=k)

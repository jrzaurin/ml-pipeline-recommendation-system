import json
from datetime import timedelta
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark.sql import Row
from random import randint

import luigi
import dask.dataframe as dd

from utils import Mario, start_spark
from x04_train_test_split import FilteredDevSet
from baselines import MostPopularReco, RandomReco, MostPopularInCatReco, ItemPopularity
from x04_train_test_split import ONE_YEAR_REVIEWS_JOB, DEV_SET_REVIEWS_JOB
from x04_train_test_split import FULL_TRAIN_USERS_JOB
from biggraph import PBGReco, PBGRecoV2
from train_set import ItemRelevance


def get_reco_job(reco_name, k):
    days = 31
    base_args = dict(
        item_days=days,
        k=k,
        dim=100,
        loss_fn='softmax',
        # comparator='l2', # this is hardcoded right now as
        # l2
        lr=0.1,
        eval_fraction=0.05,
        regularization_coef=1e-3,
        num_negs=1000,
    )
    return {
        'Most Popular': MostPopularReco(days=days, k=k),
        'Most Popular in Cat': MostPopularInCatReco(days=days, k=k),
        'Random': RandomReco(days=days, k=k),
        'PBG V01': PBGReco(
            epochs=2,
            days=2,
            min_user_rev=2,
            **base_args
        ),
        'PBG V02': PBGReco(
            epochs=50,
            days=2,
            min_user_rev=2,
            **base_args
        ),
        'PBG V03': PBGReco(
            epochs=50,
            days=365,
            min_user_rev=5,
            **base_args
        ),
        'PBG V04': PBGReco(
            epochs=200,
            days=365,
            min_user_rev=5,
            **base_args
        ),
        'PBG V05': PBGReco(
            epochs=20,
            days=365,
            min_user_rev=5,
            **base_args
        ),
        'PBG V06': PBGReco(
            epochs=50,
            days=365,
            min_user_rev=2,
            **base_args
        ),
        'PBG V07': PBGReco(
            epochs=50,
            days=365,
            min_user_rev=1,
            **base_args
        ),
        'PBG V08': PBGReco(
            epochs=100,
            days=365,
            min_user_rev=1,
            **base_args
        ),
        'PBG V2.01': PBGRecoV2(
            epochs=50,
            days=365,
            min_user_rev=5,
            min_meta_count=10,
            **base_args
        ),
        'PBG V2.02': PBGRecoV2(
            epochs=50,
            days=365,
            min_user_rev=1,
            min_meta_count=10,
            **base_args
        ),
        'PBG V2.03': PBGRecoV2(
            epochs=100,
            days=365,
            min_user_rev=1,
            min_meta_count=10,
            **base_args
        ),
        'PBG V2.04': PBGRecoV2(
            epochs=25,
            days=365,
            min_user_rev=1,
            min_meta_count=10,
            **base_args
        ),
        'PBG V2.00': PBGRecoV2(
            epochs=3,
            days=365,
            min_user_rev=1,
            min_meta_count=10,
            **base_args
        ),
        'PBG V2.05': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=50,
            days=365,
            min_user_rev=1,
            min_meta_count=10,

            dim=50,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=1000,
        ),
        'PBG V2.06': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=50,
            days=365,
            min_user_rev=1,
            min_meta_count=10,

            dim=200,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=1000,
        ),
        'PBG V2.07': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=50,
            days=365,
            min_user_rev=1,
            min_meta_count=10,

            dim=100,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=100,
        ),
        'PBG V2.08': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=50,
            days=365,
            min_user_rev=1,
            min_meta_count=10,

            dim=100,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.01,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=1000,
        ),
        'PBG V2.09': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=50,
            days=365,
            min_user_rev=3,
            min_meta_count=10,

            dim=100,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=1000,
        ),
        'PBG V2.10': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=20,
            days=365,
            min_user_rev=1,
            min_meta_count=10,

            dim=100,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=1000,
        ),
        'PBG V2.11': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=25,
            days=365,
            min_user_rev=5,
            min_meta_count=10,

            dim=200,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=100,
        ),
        'PBG V2.12': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=100,
            days=365,
            min_user_rev=5,
            min_meta_count=10,

            dim=200,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=100,
        ),
        'PBG V2.13': PBGRecoV2(
            item_days=days,
            k=k,

            epochs=200,
            days=365,
            min_user_rev=5,
            min_meta_count=10,

            dim=200,
            loss_fn='softmax',
            # comparator='l2', # this is hardcoded right now as
            # l2
            lr=0.1,
            eval_fraction=0.05,
            regularization_coef=1e-3,
            num_negs=100,
        ),
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


class ItemsToScore(Mario, luigi.Task):
    """
    set of k items per user.

    The items include all the items the user has reviewed in the DEV set
    (unless it's more than k). The rest are drawn at random from the set of
    recent items (that includes all items from dev set + last self.days days
    of the train set.

    Only warm users are included - users with at least 1 interaction in the
    extended training set (that is the last 5 years).
    """
    k = luigi.IntParameter()
    days = luigi.IntParameter()

    def output_dir(self):
        return 'dev_metrics/items_to_score/days_%s_k_%s' % (self.days, self.k)

    def requires(self):
        return [
            DevItemRelevance(),
            ItemPopularity(days=self.days),
            FULL_TRAIN_USERS_JOB
        ]

    def _run(self):
        sc, sqlc = start_spark()
        rel_job, recent_items_job, users_job = self.requires()

        dev_set_items = (
            rel_job.load_parquet(sqlc=sqlc, from_local=True)
            .select('item')
            .distinct()
        )

        recent_items = recent_items_job.load_parquet(
            sqlc=sqlc, from_local=True).select('item')
        recent_items.toPandas()
        dev_set_items.printSchema()
        recent_items.printSchema()
        total_items = dev_set_items.union(recent_items).distinct().toPandas()

        total_items['ind'] = total_items.index
        n = len(total_items)
        item2ind = sqlc.createDataFrame(total_items)

        relevance = rel_job.load_parquet(sqlc=sqlc, from_local=True)

        k = self.k

        def generate_items(x):
            user, interactions = x
            interactions = sorted(interactions, key=lambda x: -x[1])[:k]
            items = {item_ind for item_ind in interactions}
            while len(interactions) < k:
                new_item = randint(0, n - 1)
                if new_item not in items:
                    interactions.append((new_item, 0))
                    items.add(new_item)

            for i, (item_ind, item_relevance) in enumerate(interactions):
                yield Row(
                    reviewerID=user,
                    rank=i + 1,
                    ind=item_ind,
                    relevance=item_relevance
                )
        users = users_job.load_parquet(sqlc=sqlc, from_local=True)
        result = (
            relevance
            .join(users, on='reviewerID')               # limit to warm users
            # translate item id to index
            .join(item2ind, on='item')
            .select('reviewerID', 'ind', 'relevance')
            .rdd
            .map(lambda x: (x.reviewerID, [(x.ind, x.relevance)]))
            .reduceByKey(lambda a, b: a + b)
            .flatMap(generate_items)
            .toDF()
            # translate index back to item id
            .join(item2ind, on='ind')
            .select(
                'reviewerID',
                'item',
                'rank',
                'relevance'
            )
        )
        self.save_parquet(result.repartition(200))


class UserTemperature(Mario, luigi.Task):

    def requires(self):
        return ONE_YEAR_REVIEWS_JOB, DEV_SET_REVIEWS_JOB

    def output_dir(self):
        return 'dev_metrics/user_temperature'

    def _run(self):
        dev_users = DEV_SET_REVIEWS_JOB.load_parquet()[
            ['reviewerID']].drop_duplicates()
        train_reviews = ONE_YEAR_REVIEWS_JOB.load_parquet()

        counts = (
            train_reviews[['reviewerID', 'item']]
            .drop_duplicates()
            .groupby('reviewerID')
            .count()
            .reset_index()
            .rename(columns={'item': 'interactions'})
            .merge(dev_users, on='reviewerID')
        )
        self.save_parquet(counts)


class DevItemRelevance(ItemRelevance):
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
            DevItemRelevance(),
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


class EvaluateRecoDetail(Mario, luigi.Task):
    """
    NDCG: double
    DCG: double
    hits: double
    Reco: string
    k: int64
    """
    reco_name = luigi.Parameter()
    k = luigi.IntParameter()
    hot_threshold = 6

    def output_dir(self):
        return 'dev_metrics/eval2/%s/k=%s' % (self.reco_name, self.k)

    def requires(self):
        return [
            DevItemRelevance(),
            UserTemperature(),
            get_reco_job(self.reco_name, self.k)
        ]

    def _run(self):

        relevance_job, temperature_job, reco_job = self.requires()
        user_temp = temperature_job.load_parquet()
        relevance = relevance_job.load_parquet().compute()
        recs = reco_job.load_parquet().compute()

        # sanity checks
        all_users = set(relevance.reviewerID)
        assert set(recs.reviewerID) == all_users
        assert set(recs[recs['rank'] == 1].reviewerID) == all_users
        assert set(recs[recs['rank'] == self.k].reviewerID) == all_users

        all_results = []
        # calculate metrics separately for user temperature = 1, 2, ...
        for temperature in range(1, self.hot_threshold):
            users = user_temp[user_temp.interactions == temperature].compute()
            recs_subset = recs.merge(users, on='reviewerID')
            relevance_subset = relevance.merge(users, on='reviewerID')

            metrics = ndcg(recs_subset, relevance_subset, self.k)
            results = dict(metrics.mean())
            results['temperature'] = str(temperature)
            results['user_count'] = len(users)
            results_pd = pd.DataFrame([results])
            all_results.append(results_pd)

        # and the same for hot users (temp >= hot_threshold)
        users = user_temp[user_temp.interactions >=
                          self.hot_threshold].compute()
        recs_subset = recs.merge(users, on='reviewerID')
        relevance_subset = relevance.merge(users, on='reviewerID')

        metrics = ndcg(recs_subset, relevance_subset, self.k)
        results = dict(metrics.mean())
        results['temperature'] = '>= %s' % self.hot_threshold
        results['user_count'] = len(users)
        results_pd = pd.DataFrame([results])
        all_results.append(results_pd)

        final = pd.concat(all_results).reset_index(drop=True)
        final['Reco'] = self.reco_name
        final['k'] = self.k
        print(final)
        results_dd = dd.from_pandas(final, npartitions=1)
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
            'PBG V01',
            'PBG V02',
            'PBG V03',
            'PBG V04',
            'PBG V05',
            'PBG V06',
            'PBG V07',
            'PBG V08',
            'PBG V2.10',
            'PBG V2.05',
            'PBG V2.00',
            'PBG V2.01',
            'PBG V2.02',


            'PBG V2.06',
            'PBG V2.07',
            'PBG V2.04',
            'PBG V2.11',
            'PBG V2.12',
            'PBG V2.13',
            # 'PBG V2.03`',

            # 'PBG V2.08',
            # 'PBG V2.09',



        ]
        for k in [20, 40]:
            # for k in [10]:
            for name in reco_names:
                yield EvaluateRecoDetail(reco_name=name, k=k)

    def _run(self):
        experiments = [exp.load_parquet() for exp in self.requires()]
        all_together = dd.concat(experiments)
        print(all_together.compute())
        self.save_parquet(all_together)


def get_reqs(job):
    reqs = job.requires()
    if isinstance(reqs, Mario):
        return [reqs]
    else:
        return list(reqs)


def what_needs_to_run(job):
    if job.complete():
        return None

    for parent in get_reqs(job):
        what_needs = what_needs_to_run(parent)
        if what_needs is not None:
            return what_needs
    return job


def run_everything(job):
    while what_needs_to_run(job) is not None:
        what_needs_to_run(job).run()

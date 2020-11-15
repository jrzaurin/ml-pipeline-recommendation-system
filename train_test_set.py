import json
from datetime import timedelta
import random
from random import randint

import pandas as pd
import numpy as np
import luigi
import dask.bag as db
import dask.dataframe as dd
from pyspark.sql import Row

from utils import Mario, start_spark
from x04_train_test_split import n_days_subset, ONE_YEAR_REVIEWS_JOB, FULL_TRAIN_USERS_JOB
from x04_train_test_split import FilteredDevSet
from x03_parquetify import ParquetifyMetadata
from datetime import datetime


class ItemRelevance(Mario, luigi.Task):
    """
    Calculates a measure of relevance of item for a user
    based on the train set.

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
        return 'train_set/item_relevance'

    def requires(self):
        return ONE_YEAR_REVIEWS_JOB

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
        self.save_parquet(dd.from_pandas(user_items_pd, npartitions=100))


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
        return 'dev_set/item_relevance'

    def requires(self):
        return FilteredDevSet()


class ExtendedTrainSet(Mario, luigi.Task):
    """
    Set of all interactions from the training set + sampled negative feedback.
    That is

    Includes every pair (user, item) for where a user has reviewed the item.
    In addition, for every item reviewed by the user, includes n items that
    the user has NOT reviewed (with relevance = 0).

    reviewerID: string
    item: string
    relevance: int64
    """
    seed = luigi.IntParameter()
    n = luigi.IntParameter()

    def output_dir(self):
        return 'train_set/w_negatives/n=%s/seed=%s' % (self.n, self.seed)

    def requires(self):
        return ItemRelevance()

    def _run(self):
        sc, sqlc = start_spark()
        relevance = self.requires().load_parquet(sqlc=sqlc, from_local=True)

        total_items = (
            relevance
            .select('item')
            .distinct()
            .toPandas()
        )

        total_items['ind'] = total_items.index
        items_count = len(total_items)
        item2ind = sqlc.createDataFrame(total_items)

        n = self.n
        seed = self.seed

        def generate_items(x):
            user, interactions = x
            random.seed(hash(user) + seed)
            positives = len(interactions)
            items = {item_ind for item_ind in interactions}
            while len(interactions) < (n + 1) * positives:
                new_item = random.randint(0, items_count - 1)
                if new_item not in items:
                    interactions.append((new_item, 0))
                    items.add(new_item)

            for i, (item_ind, item_relevance) in enumerate(interactions):
                yield Row(
                    reviewerID=user,
                    ind=item_ind,
                    relevance=item_relevance
                )

        result = (
            relevance
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
                'relevance'
            )
        )
        self.save_parquet(result.repartition(200))


class DevSet(Mario, luigi.Task):
    """
    set of k items per user.

    The items include all the items the user has reviewed in the DEV set
    (unless it's more than k). The rest are drawn at random from the set of
    recent items (that includes all items from dev set + last self.days days
    of the train set.

    Only warm users are included - users with at least 1 interaction in the
    extended training set (that is the last 5 years).

    reviewerID: string
    item: string
    rank: int64
    relevance: int64
    """
    k = luigi.IntParameter()
    days = luigi.IntParameter()

    def output_dir(self):
        return 'dev_set/items_to_score/days_%s_k_%s' % (self.days, self.k)

    def requires(self):
        return [
            DevItemRelevance(),
            n_days_subset(self.days),
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

        recent_items = (
            recent_items_job
            .load_parquet(sqlc=sqlc, from_local=True)
            .select('item')
            .distinct()
        )
        recent_items.toPandas()
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

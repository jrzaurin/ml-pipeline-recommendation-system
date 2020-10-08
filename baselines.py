import json
from datetime import timedelta
from random import choices

import pandas as pd
import numpy as np
import luigi
import dask.bag as db
import dask.dataframe as dd

from utils import Mario
from x04_train_test_split import TRAIN_SET_REVIEWS_JOB, DEV_SET_REVIEWS_JOB
from datetime import datetime


class ItemPopularity(Mario, luigi.Task):
    """
    Calculates item popularity statistics for the last n days of the training set.
    Also finds top 1000 most popular items by number of reviews. Saves separately:

    stats
    -----------------
    __null_dask_index__: int64
    item: string
    rating_1: int64
    rating_2: int64
    rating_3: int64
    rating_4: int64
    rating_5: int64
    total_reviews: int64
    mean_rating: double


    top_1k
    ------------------
    __null_dask_index__: int64
    item: string
    rating_1: int64
    rating_2: int64
    rating_3: int64
    rating_4: int64
    rating_5: int64
    total_reviews: int64
    mean_rating: double
    """
    days = luigi.IntParameter()

    def output_dir(self):
        return 'baselines/item_popularity/%s_days' % self.days

    def requires(self):
        return TRAIN_SET_REVIEWS_JOB

    def _run(self):
        start_day = np.datetime64(
            self.requires().date_interval.date_b - timedelta(self.days)
        )
        df_full = self.requires().load_parquet()
        df = df_full[df_full.reviewDate >= start_day]
        df['total'] = 1

        # need to make it categorical so can pivot table on it
        df['overall'] = df.overall.astype(
            'int').astype('category').cat.as_known()

        stats = (
            df.groupby(['item', 'overall'])
            .agg({'total': 'sum'})
            .reset_index()
            .pivot_table(index='item', columns='overall', values='total')
            .astype({i: 'int' for i in range(1, 6)})
            .rename(columns={i: 'rating_%s' % i for i in range(1, 6)})
            .reset_index()
        )
        stats.columns.name = None

        stats['total_reviews'] = (
            stats.rating_1
            + stats.rating_2
            + stats.rating_3
            + stats.rating_4
            + stats.rating_5
        )
        stats['mean_rating'] = (
            stats.rating_1
            + 2 * stats.rating_2
            + 3 * stats.rating_3
            + 4 * stats.rating_4
            + 5 * stats.rating_5
        ) / stats.total_reviews

        self.save_parquet(stats, 'stats')

        stats = self.load_parquet('stats')
        top_1k = stats.nlargest(1000, 'total_reviews').reset_index(drop=True)
        self.save_parquet(top_1k, 'top_1k')


class UserStats(Mario, luigi.Task):
    """
    Creates a list of all users in the training set,
    calculates number of ratings per use + avg rating - because why not.

    mean_rating: double
    review_count: int64
    reviewerID: string
    """

    def output_dir(self):
        return 'baselines/user_stats'

    def requires(self):
        return TRAIN_SET_REVIEWS_JOB

    def _run(self):
        df = self.requires().load_parquet()
        agged = df.groupby(['reviewerID']).agg(
            {'overall': ['mean', 'count']}).reset_index()
        final = agged['overall']
        final['reviewerID'] = agged['reviewerID']
        final = final.rename(
            columns={
                'count': 'review_count',
                'mean': 'mean_rating'})
        self.save_parquet(final)


class DevUserStats(UserStats):
    """
    Creates a list of all users in the dev set,
    calculates number of ratings per use + avg rating - because why not.

    mean_rating: double
    review_count: int64
    reviewerID: string
    """

    def output_dir(self):
        return 'baselines/dev_user_stats'

    def requires(self):
        return DEV_SET_REVIEWS_JOB


class MostPopularReco(Mario, luigi.Task):
    """
    Creates a recommendation of k most popular items from the last k days
    for every user.

    reviewerID: string
    item: string
    rank: int64
    """
    k = luigi.IntParameter()
    days = luigi.IntParameter()

    def output_dir(self):
        return 'baselines/most_popular/k=%s_days=%s' % (self.k, self.days)

    def requires(self):
        return [ItemPopularity(days=self.days), UserStats(), DevUserStats()]

    def _run(self):
        item_pop_job, user_stats_job, dev_users_job = self.requires()
        dev_users = dev_users_job.load_parquet()[['reviewerID']]

        top_items = (
            item_pop_job
            .load_parquet('top_1k')
            .nlargest(self.k, 'total_reviews')[['item', 'total_reviews']]
            .compute()
        )
        top_items['rank'] = top_items.index + 1

        users = (
            UserStats().load_parquet()
            [['reviewerID']]
            .merge(dev_users, on='reviewerID')
        )
        users['key'] = 1
        top_items['key'] = 1

        recommendations = users.merge(top_items, on='key')[
            ['reviewerID', 'item', 'rank']]
        self.save_parquet(recommendations.repartition(partition_size='50MB'))


class RandomReco(Mario, luigi.Task):
    """
    Creates a recommendation of k random items from last month
    for every user.

    reviewerID: string
    item: string
    rank: int64
    """
    k = luigi.IntParameter()
    days = luigi.IntParameter()

    def output_dir(self):
        return 'baselines/random/k=%s_days=%s' % (self.k, self.days)

    def requires(self):
        return [ItemPopularity(days=self.days), UserStats(), DevUserStats()]

    def _run(self):
        item_pop_job, user_stats_job, dev_users_job = self.requires()
        dev_users = dev_users_job.load_parquet()[['reviewerID']]

        all_items = item_pop_job.load_parquet('stats')[['item']].compute().item

        users = (
            UserStats().load_parquet()
            [['reviewerID']]
            .merge(dev_users, on='reviewerID')
            .compute()
            .reviewerID
        )

        reco = pd.DataFrame({
            'reviewerID': np.hstack([users] * self.k),
            'item': choices(all_items, k=len(users) * self.k),
            'rank': np.hstack([[i] * len(users) for i in range(1, self.k + 1)])
        })
        self.save_parquet(dd.from_pandas(reco, chunksize=50000))

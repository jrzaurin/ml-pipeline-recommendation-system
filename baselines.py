import json
from datetime import timedelta
from random import choices

import pandas as pd
import numpy as np
import luigi
import dask.bag as db
import dask.dataframe as dd

from utils import Mario, start_spark
from x04_train_test_split import n_days_subset, FilteredDevSet
from x03_parquetify import ParquetifyMetadata
from datetime import datetime


class ItemPopularity(Mario, luigi.Task):
    """
    Calculates item popularity statistics for the last n days of the training set.
    Also finds top 1000 most popular items by number of reviews. Saves separately:

    item: string
    rating_1: int64
    rating_2: int64
    rating_3: int64
    rating_4: int64
    rating_5: int64
    cat_1: string
    cat_2: string
    cat_3: string
    cat_4: string
    total_reviews: int64
    mean_rating: double
    rank: int64
    rank_in_cat_1: int64
    """
    days = luigi.IntParameter()

    def output_dir(self):
        return 'baselines/item_popularity/%s_days' % self.days

    def requires(self):
        return n_days_subset(self.days), ParquetifyMetadata()

    def _run(self):
        # truncate category names longer than this
        CHAR_LIMIT = 100
        reviews_job, meta_job = self.requires()

        meta = meta_job.load_parquet()
        meta['cat_1'] = meta.category.map(lambda x: x[1])
        meta['cat_2'] = meta.category.map(
            lambda x: x[2][:CHAR_LIMIT] if len(x) > 2 else '')
        meta['cat_3'] = meta.category.map(
            lambda x: x[3][:CHAR_LIMIT] if len(x) > 3 else '')
        meta['cat_4'] = meta.category.map(
            lambda x: x[4][:CHAR_LIMIT] if len(x) > 4 else '')

        meta = meta[['item', 'cat_1', 'cat_2', 'cat_3', 'cat_4']
                    ].groupby('item').first().reset_index()

        start_day = str(
            reviews_job.date_interval.date_b - timedelta(self.days)
        )
        df = reviews_job.load_parquet()
        df['total'] = 1

        # need to make it categorical so can pivot table on it
        df['overall'] = df.overall.astype(
            'int').astype('category').cat.as_known()

        merged = (
            df.groupby(['item', 'overall'])
            .agg({'total': 'sum'})
            .reset_index()
            .pivot_table(index='item', columns='overall', values='total')
            .astype({i: 'int' for i in range(1, 6)})
            .rename(columns={i: 'rating_%s' % i for i in range(1, 6)})
            .reset_index()
            .merge(meta, on='item', how='left')
            .compute()
        )
        merged.fillna('', inplace=True)

        merged['total_reviews'] = (
            merged.rating_1
            + merged.rating_2
            + merged.rating_3
            + merged.rating_4
            + merged.rating_5
        )
        merged['mean_rating'] = (
            merged.rating_1
            + 2 * merged.rating_2
            + 3 * merged.rating_3
            + 4 * merged.rating_4
            + 5 * merged.rating_5
        ) / merged.total_reviews

        merged = merged.sort_values(
            'total_reviews',
            ascending=False).reset_index(
            drop=True)
        merged['rank'] = merged.index + 1
        merged['rank_in_cat_1'] = (
            merged
            .groupby('cat_1')
            .total_reviews
            .rank(
                method='first', ascending=False
            )
        ).astype('int')

        self.save_parquet(
            dd.from_pandas(
                merged,
                npartitions=1).repartition(
                partition_size='50mb'))


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


# TODO: this is hardcoded to cover all of training period - make it not
# hardcoded
TRAIN_SET_ITEM_STATS = ItemPopularity(days=1825)

ONE_YEAR_ITEM_STATS = ItemPopularity(days=365)


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
        return FilteredDevSet()


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
        return [ItemPopularity(days=self.days), DevUserStats()]

    def _run(self):
        item_pop_job, dev_users_job = self.requires()
        users = dev_users_job.load_parquet()[['reviewerID']]

        items = item_pop_job.load_parquet()
        top_items = items[items['rank'] <= self.k][['item', 'rank']]

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
        return [ItemPopularity(days=self.days), DevUserStats()]

    def _run(self):
        item_pop_job, dev_users_job = self.requires()

        users = dev_users_job.load_parquet()[
            ['reviewerID']].reviewerID.compute()
        all_items = item_pop_job.load_parquet()[['item']].item.compute()

        reco = pd.DataFrame({
            'reviewerID': np.hstack([users] * self.k),
            'item': choices(all_items, k=len(users) * self.k),
            'rank': np.hstack([[i] * len(users) for i in range(1, self.k + 1)])
        })
        self.save_parquet(dd.from_pandas(reco, chunksize=50000))


class UserFavCat(Mario, luigi.Task):
    """
    Favourite item category per user based on last year of reviews.

    reviewerID: string
    cat_1: string
    cat_count: int6
    """

    def output_dir(self):
        return 'baselines/user_fav_cat/1_year'

    def requires(self):
        return [
            ItemPopularity(days=365),
            n_days_subset(365)
        ]

    def _run(self):
        item_pop_job, reviews_job = self.requires()
        items = item_pop_job.load_parquet()

        user_cat_counts = (
            reviews_job.load_parquet()
            .merge(items[['item', 'cat_1']], on='item')
            [['item', 'reviewerID', 'cat_1']]
            .groupby(['reviewerID', 'cat_1'])
            .count()
            .reset_index()
            .rename(columns={'item': 'cat_count'})
        )

        user_cat_counts['max_cat_count'] = (
            user_cat_counts
            .groupby(['reviewerID'])
            ['cat_count'].transform(max, meta='int64')
        )

        result = (
            user_cat_counts[user_cat_counts.cat_count == user_cat_counts.max_cat_count]
            .groupby('reviewerID')
            .first()
            .reset_index()
        )

        self.save_parquet(result[['reviewerID', 'cat_1', 'cat_count']].repartition(
            partition_size='20MB'))


class MostPopularInCatReco(Mario, luigi.Task):
    """
    Creates a recommendation of k most populars items from last month
    for every user.

    cat_1: string not null
    reviewerID: string
    item: string
    rank: int64
    """
    k = luigi.IntParameter()
    days = luigi.IntParameter()

    def output_dir(self):
        return 'baselines/most_pop_in_cat/k=%s_days=%s' % (self.k, self.days)

    def requires(self):
        return UserFavCat(), ItemPopularity(days=self.days), DevUserStats()

    def _run(self):
        sc, sqlc = start_spark()
        user_cats_job, items_job, dev_users_job = self.requires()
        fav_cats = user_cats_job.load_parquet(sqlc=sqlc).filter('cat_1 != ""')
        test_users = dev_users_job.load_parquet(sqlc=sqlc).select('reviewerID')

        UNKNOWN_CAT = 'UNKNOWN FAVORITE CATEGORY'
        full_fav_cats = (
            test_users
            .join(fav_cats, on='reviewerID', how='left')
            .selectExpr(
                'reviewerID',
                'coalesce(cat_1, "%s") as cat_1' % UNKNOWN_CAT
            )
        )

        top_items_overall = (
            items_job.load_parquet(sqlc=sqlc)
            .filter('rank <= %s' % self.k)
            .selectExpr(
                'item',
                '"%s" as cat_1' % UNKNOWN_CAT,
                'rank'
            )
        )
        items_per_cat = (
            items_job.load_parquet(sqlc=sqlc)
            .filter('rank_in_cat_1 <= %s' % self.k)
            .selectExpr(
                'item',
                'cat_1',
                'rank_in_cat_1 as rank'
            )
            .unionAll(top_items_overall)
        )
        result = full_fav_cats.join(items_per_cat, on='cat_1')
        self.save_parquet(result)

import json
from datetime import date, timedelta

import pandas as pd
import luigi
import dask.bag as db

from utils import Mario, start_spark
from x03_parquetify import ParquetifyReviews


class SubsetReviews(Mario, luigi.Task):
    """
    overall: double
    vote: string
    verified: bool
    reviewerID: string
    item: string
    style: list<item: list<item: string>>
      child 0, item: list<item: string>
          child 0, item: string
    reviewerName: string
    reviewText: string
    summary: string
    reviewDate: date32[day]
    """
    date_interval = luigi.DateIntervalParameter()

    def requires(self):
        return ParquetifyReviews()

    def output_dir(self):
        return 'reviews_subset/%s' % self.date_interval

    def _run(self):
        start = self.date_interval.date_a
        end = self.date_interval.date_b

        df = self.requires().load_parquet()
        cond1 = df.reviewDate.map(
            lambda x: x >= start, meta=pd.Series(
                [], dtype=bool))
        cond2 = df.reviewDate.map(
            lambda x: x <= end, meta=pd.Series(
                [], dtype=bool))
        filtered = df[cond1 & cond2]
        filtered.to_parquet(self.full_output_dir())


def n_days_subset(n):
    """training set job limited to the last n days"""
    end_date = date(2018, 3, 31)
    start_date = end_date - timedelta(n - 1)
    return SubsetReviews(date_interval=luigi.date_interval.Custom(
        start_date,
        end_date
    ))


MINI_REVIEWS_JOB = n_days_subset(2)
ONE_YEAR_REVIEWS_JOB = n_days_subset(365)
FULL_TRAIN_SET_REVIEWS_JOB = SubsetReviews(
    date_interval=luigi.date_interval.Custom(
        date(2013, 4, 1),
        date(2018, 3, 31)
    )
)

DEV_SET_REVIEWS_JOB = SubsetReviews(
    date_interval=luigi.date_interval.Custom(
        date(2018, 4, 1),
        date(2018, 4, 30)
    )
)
TEST_SET_REVIEWS_JOB = SubsetReviews(
    date_interval=luigi.date_interval.Custom(
        date(2018, 5, 1),
        date(2018, 5, 31)
    )
)


class FilteredDevSet(Mario, luigi.Task):
    def output_dir(self):
        return 'reviews_subset/dev_set_filtered'

    def requires(self):
        return FULL_TRAIN_SET_REVIEWS_JOB, DEV_SET_REVIEWS_JOB

    def _run(self):
        sc, sqlc = start_spark()
        train_job, dev_job = self.requires()
        train_set_users = train_job.load_parquet(
            sqlc=sqlc).select('reviewerID').distinct()
        dev_set = dev_job.load_parquet(sqlc=sqlc)
        self.save_parquet(
            dev_set
            .join(train_set_users, on='reviewerID')
            .repartition(100)
        )

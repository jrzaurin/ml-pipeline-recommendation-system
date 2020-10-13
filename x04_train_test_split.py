import json
from datetime import date

import pandas as pd
import luigi
import dask.bag as db

from utils import Mario
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


MINI_REVIEWS_JOB = SubsetReviews(
    date_interval=luigi.date_interval.Custom(
        date(2018, 3, 25),
        date(2018, 3, 31)
    )
)

ONE_YEAR_REVIEWS_JOB = SubsetReviews(
    date_interval=luigi.date_interval.Custom(
        date(2017, 4, 1),
        date(2018, 3, 31)
    )
)

TRAIN_SET_REVIEWS_JOB = SubsetReviews(
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


class TrainDevTestSplit(Mario, luigi.Task):
    def output_dir(self):
        return 'junk/train_dev_test_sink/v1'

    def requires(self):
        return [
            TRAIN_SET_REVIEWS_JOB,
            DEV_SET_REVIEWS_JOB,
            TEST_SET_REVIEWS_JOB]

    def _run(self):
        pass

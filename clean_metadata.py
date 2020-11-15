import json
from datetime import timedelta
from random import choices

import pandas as pd
import numpy as np
import luigi
import dask.bag as db
import dask.dataframe as dd
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

from utils import Mario, start_spark
from x04_train_test_split import n_days_subset, FilteredDevSet
from x03_parquetify import ParquetifyMetadata
from datetime import datetime


class OnlyShortMeta(Mario, luigi.Task):
    """
    item: string
    brand: string
    category_fixed: list<element: string>
    child 0, element: string
    feature_fixed: list<element: string>
    child 0, element: string
    """

    def output_dir(self):
        return 'clean_metadata/v1'

    def requires(self):
        return ParquetifyMetadata()

    def _run(self):
        sc, sqlc = start_spark()
        meta_job = self.requires()
        meta_job.sync_output_to_local()

        df = meta_job.load_parquet(sqlc=sqlc, from_local=True)

        def filter_array(x):
            return [
                elem.strip() for elem in x
                if len(elem) < 60
                and not elem.strip().startswith('<')]

        def fix_cat_array(x):
            return filter_array(x[1:])

        array_fixer_udf = udf(filter_array, ArrayType(StringType()))
        cat_fixer = udf(fix_cat_array, ArrayType(StringType()))

        self.save_parquet(
            df.select(
                'item',
                'brand',
                cat_fixer('category').alias('category_fixed'),
                array_fixer_udf('feature').alias('feature_fixed')
            )
            .rdd
            .keyBy(lambda x: x.item)
            .reduceByKey(lambda a, b: a)
            .values()
            .toDF()
            .repartition(200)
        )

        df = self.load_parquet(sqlc=sqlc, from_local=True)
        assert df.count() == df.select('item').distinct().count()


class CatNormedMeta(Mario, luigi.Task):
    def output_dir(self):
        return 'clean_metadata/v2'

    def requires(self):
        return OnlyShortMeta()

    def _run(self):
        sc, sqlc = start_spark()

        meta = self.requires().load_parquet(sqlc=sqlc, from_local=True)

        cat1_udf = udf(lambda x: x[0], StringType())
        cat2_udf = udf(lambda x: '' if len(x) < 2 else x[1], StringType())
        cat3_udf = udf(lambda x: '' if len(x) < 3 else x[2], StringType())
        cat4_udf = udf(lambda x: '' if len(x) < 4 else x[3], StringType())
        cat5_udf = udf(lambda x: '' if len(x) < 5 else x[4], StringType())

        self.save_parquet(
            meta
            .select(
                cat1_udf('category_fixed').alias('cat_1'),
                cat2_udf('category_fixed').alias('cat_2'),
                cat3_udf('category_fixed').alias('cat_3'),
                cat4_udf('category_fixed').alias('cat_4'),
                cat5_udf('category_fixed').alias('cat_5'),
                'brand',
                'feature_fixed',
                'item'
            )
        )

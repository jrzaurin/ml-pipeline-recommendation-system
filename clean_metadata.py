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
        from pyspark.sql.functions import udf
        from pyspark.sql.types import ArrayType, StringType

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
            .repartition(200)
        )

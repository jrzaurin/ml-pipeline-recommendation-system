
import luigi
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from clean_metadata import OnlyShortMeta
from x04_train_test_split import ONE_YEAR_REVIEWS_JOB
from utils import Mario, start_spark


class ItemMetaCountEncoded(Mario, luigi.Task):
    """
    args:
    train_set: bool

    Returns total count of interactions by all users for a given
    item, brand, cat_1, cat_2, cat_3, cat_4, cat_5.

    If train_set - calculates in a leave-one-out fashion (meaning - subtracts 1 from all counts)

    Indexed by item.
    pyarrow.Table
    item: string
    brand: string
    cat_5: string
    cat_4: string
    cat_3: string
    cat_2: string
    cat_1: string
    cat_1_ce: int64 not null
    cat_2_ce: int64 not null
    cat_3_ce: int64 not null
    cat_4_ce: int64 not null
    cat_5_ce: int64 not null
    brand_ce: int64 not null
    item_ce: int64 not null
    """

    train_set = luigi.BoolParameter()

    def output_dir(self):
        return 'item_features/count_encoded/%s' % (
            'train' if self.train_set else 'test')

    def requires(self):
        return OnlyShortMeta(), ONE_YEAR_REVIEWS_JOB

    def _run(self):
        meta_job, reviews_job = self.requires()
        sc, sqlc = start_spark()

        meta = meta_job.load_parquet(sqlc=sqlc, from_local=True)
        reviews = reviews_job.load_parquet(sqlc=sqlc, from_local=True)
        relevant_items = (
            reviews_job.load_parquet(sqlc=sqlc, from_local=True)
            .select('item')
            .distinct()
        )

        meta1 = meta.join(relevant_items, on='item').cache()

        cat1_udf = udf(lambda x: x[0], StringType())
        cat2_udf = udf(lambda x: '' if len(x) < 2 else x[1], StringType())
        cat3_udf = udf(lambda x: '' if len(x) < 3 else x[2], StringType())
        cat4_udf = udf(lambda x: '' if len(x) < 4 else x[3], StringType())
        cat5_udf = udf(lambda x: '' if len(x) < 5 else x[4], StringType())

        meta2 = (
            meta1
            .select(
                cat1_udf('category_fixed').alias('cat_1'),
                cat2_udf('category_fixed').alias('cat_2'),
                cat3_udf('category_fixed').alias('cat_3'),
                cat4_udf('category_fixed').alias('cat_4'),
                cat5_udf('category_fixed').alias('cat_5'),
                'brand',
                'item'
            )
        )
        cols_to_encode = [
            'cat_1',
            'cat_2',
            'cat_3',
            'cat_4',
            'cat_5',
            'brand',
            'item'
        ]
        reviews_meta = reviews.join(meta2, on='item').cache()
        features = meta2
        for col in cols_to_encode:
            feat_name = 'f_' + col + '_ce'
            count_df = (
                reviews_meta
                .groupBy(col)
                .count()
                .withColumnRenamed('count', feat_name)
            )
            features = features.join(count_df, on=col)

        if self.train_set:
            self.save_parquet(
                features.selectExpr(
                    'item',
                    'brand',
                    'cat_5',
                    'cat_4',
                    'cat_3',
                    'cat_2',
                    'cat_1',
                    'f_cat_1_ce - 1 as f_cat_1_ce',
                    'f_cat_2_ce - 1 as f_cat_2_ce',
                    'f_cat_3_ce - 1 as f_cat_3_ce',
                    'f_cat_4_ce - 1 as f_cat_4_ce',
                    'f_cat_5_ce - 1 as f_cat_5_ce',
                    'f_brand_ce - 1 as f_brand_ce',
                    'f_item_ce - 1 as f_item_ce'
                )
                .filter('f_item_ce > 0')
            )
        else:
            self.save_parquet(features)

        df = self.load_parquet(sqlc=sqlc, from_local=True)
        assert df.count() == df.select('item').count()

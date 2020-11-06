
import luigi
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from clean_metadata import OnlyShortMeta
from x04_train_test_split import ONE_YEAR_REVIEWS_JOB
from utils import Mario, start_spark


class ItemMetaCountEncoded(Mario, luigi.Task):
    """
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

    def output_dir(self):
        return 'item_features/count_encoded'

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
            count_df = (
                reviews_meta
                .groupBy(col)
                .count()
                .withColumnRenamed('count', col + '_ce')
            )
            features = features.join(count_df, on=col)

        self.save_parquet(features)


import luigi
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from clean_metadata import CatNormedMeta
from x04_train_test_split import ONE_YEAR_REVIEWS_JOB
from utils import Mario, start_spark


class MetaInteractionCounts(Mario, luigi.Task):
    """
    args:
    smoothing: float
    train_set: bool

    Calculates the affinity of a given user for a given brand, (sub)category
    or item. This is related to target encoding.

    affinity(user, some brand) =
        (count of reviews by this user of this brand)
        -------------------------------------------------------------
        total count of reviews by this user + smoothing constant


    And analogous for cat_1, cat_2, ... cat_5, item

    For the training set (i.e. when self.train_set == True), affinity is calculated in
    leave-one-out fashion. The modified formula is:

    affinity(user, some brand) =
        if the user only reviewed this item at most 1 time -> 0
        otherwise:

        (count of reviews by this user of this brand - 1)
        -------------------------------------------------------------
        total count of reviews by this user - 1 + smoothing constant

    Saves the result for every pair (user, item) where the affinity is > 0.

    reviewerID: string
    item: string
    f_item_te: double
    brand: string
    f_brand_te: double
    cat_5: string
    f_cat_5_te: double
    cat_4: string
    f_cat_4_te: double
    cat_3: string
    f_cat_3_te: double
    cat_2: string
    f_cat_2_te: double
    cat_1: string
    f_cat_1_te: double
    feature_fixed: list<element: string>
        child 0, element: string
    """
    smoothing = luigi.FloatParameter()
    train_set = luigi.BoolParameter()

    def output_dir(self):
        return 'user_item_features/meta_counts/smoothing_%s_%s' % (
            self.smoothing, 'train' if self.train_set else 'test')

    def requires(self):
        return CatNormedMeta(), ONE_YEAR_REVIEWS_JOB

    def _run(self):
        sc, sqlc = start_spark()

        meta_job, reviews_job = self.requires()

        reviews = reviews_job.load_parquet(sqlc=sqlc, from_local=True)
        meta = meta_job.load_parquet(sqlc=sqlc, from_local=True)

        df = (
            reviews
            .join(meta, on='item')
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

        interaction_counts = (
            reviews
            .groupBy('reviewerID')
            .count()
            .withColumnRenamed('count', 'user_int_count')
            .cache()
        )

        result = (
            reviews
            .select('reviewerID', 'item')
            .distinct()
            .join(meta, on=['item'])
            .cache()
        )

        if self.train_set:
            for col in cols_to_encode:
                result = (
                    df .groupBy(
                        [
                            'reviewerID',
                            col]).count() .withColumnRenamed(
                        'count',
                        'int_count') .join(
                        interaction_counts,
                        on='reviewerID') .filter('int_count > 1') .selectExpr(
                        'reviewerID',
                        col,
                        '(int_count - 1) / (user_int_count - 1 + {SMOOTH}) as {FEAT_NAME}'.format(
                            SMOOTH=self.smoothing,
                            FEAT_NAME='f_' +
                            col +
                            '_te')) .join(
                                result,
                                on=[
                                    'reviewerID',
                                    col]))
        else:
            for col in cols_to_encode:
                result = (
                    df .groupBy(
                        [
                            'reviewerID',
                            col]).count() .withColumnRenamed(
                        'count',
                        'int_count') .join(
                        interaction_counts,
                        on='reviewerID') .selectExpr(
                        'reviewerID',
                        col,
                        'int_count / (user_int_count + {SMOOTH}) as {FEAT_NAME}'.format(
                            SMOOTH=self.smoothing,
                            FEAT_NAME='f_' +
                            col +
                            '_te')) .join(
                                result,
                                on=[
                                    'reviewerID',
                                    col]))

        self.save_parquet(result)

        df = self.load_parquet(sqlc=sqlc, from_local=True)
        assert df.count() == df.select('item', 'reviewerID').count()

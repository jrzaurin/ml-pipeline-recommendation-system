
import luigi
import dask.dataframe as dd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from utils import Mario, start_spark
from item_features import ItemMetaCountEncoded
from user_item_features import MetaInteractionCounts
from train_test_set import ExtendedTrainSet, DevSet


def get_features_jobs(features_name: str, train_set: bool):
    """Takes a string - name of the predefined set of features and returns
    3 jobs - one with item features, one with item-user features and one with
    user based features. The jobs can be None.

    Second arg - 'train_set' - indicates is the features are calculated for the training
    or the test set - as the logic may be different.
    """
    return {
        'basic': [
            ItemMetaCountEncoded(
                train_set=train_set), MetaInteractionCounts(
                smoothing=1, train_set=train_set), None], 'basic s4': [
                    ItemMetaCountEncoded(
                        train_set=train_set), MetaInteractionCounts(
                            smoothing=4, train_set=train_set), None], 'basic s025': [
                                ItemMetaCountEncoded(), MetaInteractionCounts(
                                    smoothing=0.25), None]}[features_name]


def get_model(model_string):
    models = {
        'xgb': XGBRegressor(),
        'linreg': LinearRegression(),
    }
    return models[model_string]


def join_feats(u_i_job, i_feats_job, u_i_feats_job, u_feats_job):
    sc, sqlc = start_spark()
    data = u_i_job.load_parquet(sqlc=sqlc, from_local=True)

    def drop_bad_cols(df):
        good_cols = [
            c for c in df.columns
            if c in ['item', 'reviewerID'] or c.startswith('f_')]
        return df.select(good_cols)

    if i_feats_job is not None:
        item_feats = drop_bad_cols(
            i_feats_job.load_parquet(sqlc=sqlc, from_local=True)
        )
        data = (
            data
            .join(item_feats, on='item', how='left')
            .na.fill(0)
        )

    if u_i_feats_job is not None:
        user_item_feats = drop_bad_cols(
            u_i_feats_job.load_parquet(sqlc=sqlc, from_local=True)
        )
        data = (
            data
            .join(user_item_feats, on=['reviewerID', 'item'], how='left')
            .na.fill(0)
        )

    if u_feats_job is not None:
        user_feats = drop_bad_cols(
            u_feats_job.load_parquet(sqlc=sqlc, from_local=True)
        )
        data = (
            data
            .join(item_feats, on='reviewerID', how='left')
            .na.fill(0)
        )

    return data


class TrainFeaturesJoined(Mario, luigi.Task):
    """
    pyarrow.Table
    reviewerID: string
    item: string
    relevance: int64 not null
    f_cat_1_ce: int64 not null
    f_cat_2_ce: int64 not null
    f_cat_3_ce: int64 not null
    f_cat_4_ce: int64 not null
    f_cat_5_ce: int64 not null
    f_brand_ce: int64 not null
    f_item_ce: int64 not null
    f_item_te: double not null
    f_brand_te: double not null
    f_cat_5_te: double not null
    f_cat_4_te: double not null
    f_cat_3_te: double not null
    f_cat_2_te: double not null
    f_cat_1_te: double not null
    """
    feat_name = luigi.Parameter()
    train_n = luigi.IntParameter()
    train_seed = luigi.IntParameter()

    def output_dir(self):
        return 'tabular/train_features/%s/%s_%s' % (
            self.feat_name, self.train_n, self.train_seed
        )

    def requires(self):
        reqs = [
            ExtendedTrainSet(n=self.train_n, seed=self.train_seed)
        ]
        for job in get_features_jobs(self.feat_name, train_set=True):
            if job is not None:
                reqs.append(job)
        return reqs

    def _run(self):
        self.save_parquet(join_feats(
            self.requires()[0],
            *get_features_jobs(self.feat_name, train_set=True)
        ))


class TestFeaturesJoined(Mario, luigi.Task):
    """
    reviewerID: string
    item: string
    rank: int64 not null
    relevance: int64 not null
    f_cat_1_ce: int64 not null
    f_cat_2_ce: int64 not null
    f_cat_3_ce: int64 not null
    f_cat_4_ce: int64 not null
    f_cat_5_ce: int64 not null
    f_brand_ce: int64 not null
    f_item_ce: int64 not null
    f_item_te: double not null
    f_brand_te: double not null
    f_cat_5_te: double not null
    f_cat_4_te: double not null
    f_cat_3_te: double not null
    f_cat_2_te: double not null
    f_cat_1_te: double not null
    """
    feat_name = luigi.Parameter()
    test_k = luigi.IntParameter()
    days = luigi.IntParameter()

    def output_dir(self):
        return 'tabular/test_features/%s/%s_%s' % (
            self.feat_name, self.days, self.test_k)

    def requires(self):
        reqs = [
            DevSet(k=self.test_k, days=self.days)
        ]
        for job in get_features_jobs(self.feat_name, train_set=False):
            if job is not None:
                reqs.append(job)
        return reqs

    def _run(self):
        self.save_parquet(join_feats(
            self.requires()[0],
            *get_features_jobs(self.feat_name, train_set=False)
        ))


class TrainTestModel(Mario, luigi.Task):
    """
    saves two outputs: 'train' and 'test'
    both with the same schema:

    reviewerID: string
    item: string
    relevance: int64
    f_cat_1_ce: int64
    f_cat_2_ce: int64
    f_cat_3_ce: int64
    f_cat_4_ce: int64
    f_cat_5_ce: int64
    f_brand_ce: int64
    f_item_ce: int64
    f_item_te: double
    f_brand_te: double
    f_cat_5_te: double
    f_cat_4_te: double
    f_cat_3_te: double
    f_cat_2_te: double
    f_cat_1_te: double
    prediction: double
    """
    test_k = luigi.IntParameter()
    days = 31
    model_string = luigi.Parameter()
    feat_name = luigi.Parameter()
    train_n = luigi.IntParameter()
    train_seed = 1

    def output_dir(self):
        return 'tabular/train_test/{FEAT_NAME}/{TRAIN_N}/{MODEL}/{K}'.format(
            FEAT_NAME=self.feat_name,
            TRAIN_N=self.train_n,
            MODEL=self.model_string,
            K=self.test_k
        )

    def requires(self):
        return [
            TrainFeaturesJoined(
                feat_name=self.feat_name,
                train_n=self.train_n,
                train_seed=self.train_seed
            ),
            TestFeaturesJoined(
                feat_name=self.feat_name,
                test_k=self.test_k,
                days=self.days
            )
        ]

    def _run(self):
        model = get_model(self.model_string)

        df_train = self.requires()[0].load_parquet().compute()
        y_train = df_train.relevance
        X_train = df_train[[
            c for c in df_train.columns
            if c.startswith('f_')
        ]]

        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        df_train['prediction'] = train_preds

        self.save_parquet(dd.from_pandas(df_train, npartitions=200), 'train')

        del df_train
        del X_train
        del y_train

        df_test = self.requires()[1].load_parquet().compute()
        y_test = df_test.relevance
        X_test = df_test[[
            c for c in df_test.columns
            if c.startswith('f_')
        ]]

        test_preds = model.predict(X_test)
        df_test['prediction'] = test_preds

        self.save_parquet(dd.from_pandas(df_test, npartitions=200), 'test')

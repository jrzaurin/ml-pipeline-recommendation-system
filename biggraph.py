from pathlib import Path
import luigi
import dask.dataframe as dd
import pandas as pd
import numpy as np
import faiss
import luigi
import json
import h5py
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging
from pyspark.sql.functions import arrays_zip, col, explode

from utils import Mario, start_spark
from x04_train_test_split import n_days_subset, FilteredDevSet
from baselines import ItemPopularity, MostPopularInCatReco
from clean_metadata import OnlyShortMeta


class MakeEdgeList(Mario, luigi.Task):
    """
    saves edges.tsv - edge list for the interactions graph
    and a dataframe of all users under
    'users'
    """
    days = luigi.IntParameter()
    min_user_rev = luigi.IntParameter()

    def output_dir(self):
        return 'graph/edge_list/days_%s_min_rev_%s' % (
            self.days, self.min_user_rev)

    def requires(self):
        return n_days_subset(self.days)

    def _run(self):
        reviews = self.requires().load_parquet()[
            ['reviewerID', 'item']]
        review_counts = reviews.groupby('reviewerID').count().reset_index()
        users_subset = review_counts[review_counts.item >= self.min_user_rev][
            ['reviewerID']]

        # self.save_parquet(users_subset, 'users')

        reviews1 = reviews.merge(users_subset, on='reviewerID').compute()
        print('\n\nfound %s edges matching criteria\n\n' % len(reviews1))

        graph_path = self.local_path('edges.tsv')
        with open(graph_path, 'w') as f:
            for reviewer, item in zip(reviews1.reviewerID, reviews1.item):
                f.write(reviewer + '\tr\t' + item + '\n')

        self.backup_local_dir()


class PreparePBGInput(Mario, luigi.Task):
    days = luigi.IntParameter()
    min_user_rev = luigi.IntParameter()

    def pbg_config(self):
        return dict(
            # I/O data
            entity_path=str(self.local_path()),
            edge_paths=[
                str(self.local_path() / 'edges_partitioned'),
            ],
            checkpoint_path=str(
                self.local_path() /
                'PLACEHOLDER'),
            # placeholder
            # Graph structure
            entities={
                "user": {"num_partitions": 1},
                "item": {"num_partitions": 1}
            },
            relations=[
                {
                    "name": "r",
                    "lhs": "user",
                    "rhs": "item",
                    "operator": "none",
                }
            ],
            # ALL THIS STUFF BELOW IS JUST A PLACEHOLDER
            # IT WILL BE OVERWRITTEN DOWNSTREAM
            dynamic_relations=False,
            dimension=100,
            global_emb=False,
            comparator="dot",
            num_epochs=7,
            num_uniform_negs=1000,
            loss_fn="softmax",
            lr=0.1,
            regularization_coef=1e-3,
            eval_fraction=0.,
        )

    def output_dir(self):
        return 'graph/pbg_input/days_%s_min_rev_%s' % (
            self.days, self.min_user_rev)

    def requires(self):
        return MakeEdgeList(days=self.days, min_user_rev=self.min_user_rev)

    def _run(self):
        setup_logging()
        config = parse_config(self.pbg_config())
        subprocess_init = SubprocessInitializer()
        input_edge_paths = [self.requires().get_local_output('edges.tsv')]

        convert_input_data(
            config.entities,
            config.relations,
            config.entity_path,
            config.edge_paths,
            input_edge_paths,
            TSVEdgelistReader(lhs_col=0, rel_col=1, rhs_col=2),
            dynamic_relations=config.dynamic_relations,
        )

        self.backup_local_dir()


class TrainPBG(Mario, luigi.Task):
    """Trains embeddings. Saves

    users
    embedding: list<item: float>
        child 0, item: float
    reviewerID: string

    items
    embedding: list<item: float>
        child 0, item: float
    item: string
    """
    epochs = luigi.IntParameter()
    dim = luigi.IntParameter()
    loss_fn = luigi.Parameter()
    comparator = luigi.Parameter()
    lr = luigi.FloatParameter()
    eval_fraction = luigi.FloatParameter()
    regularization_coef = luigi.FloatParameter()
    num_negs = luigi.IntParameter()
    days = luigi.IntParameter()
    min_user_rev = luigi.IntParameter()

    def output_dir(self):
        return 'graph/train_pbg/days_{DAYS}_min_rev_{MIN_REV}/' \
            '{DIM}_{LOSS_FN}_{COMPARATOR}_{LR}_{REG_COEF}_{NUM_NEGS}_' \
            '{EVAL_FR}_{EPOCHS}'.format(
                DIM=self.dim,
                LOSS_FN=self.loss_fn,
                COMPARATOR=self.comparator,
                LR=self.lr,
                REG_COEF=self.regularization_coef,
                EVAL_FR=self.eval_fraction,
                EPOCHS=self.epochs,
                NUM_NEGS=self.num_negs,
                DAYS=self.days,
                MIN_REV=self.min_user_rev
            )

    def requires(self):
        return PreparePBGInput(days=self.days, min_user_rev=self.min_user_rev)

    def pbg_config(self):
        config = self.requires().pbg_config()
        config.update(dict(
            checkpoint_path=str(self.local_path() / 'checkpoints'),
            dimension=self.dim,
            global_emb=False,
            comparator=self.comparator,
            num_epochs=self.epochs,
            num_uniform_negs=self.num_negs,
            loss_fn=self.loss_fn,
            lr=self.lr,
            regularization_coef=self.regularization_coef,
            eval_fraction=self.eval_fraction
        ))
        return config
    # entity name, subdirectory name, column name
    entities = [
        ('user', 'users', 'reviewerID'),
        ('item', 'items', 'item')
    ]

    def _run(self):
        prep_job = self.requires()
        prep_job.sync_output_to_local()

        setup_logging()
        config = parse_config(self.pbg_config())
        subprocess_init = SubprocessInitializer()
        train(config, subprocess_init=subprocess_init)
        self.backup_local_dir()

        for entity, output_name, col_name in self.entities:

            entity_path = prep_job.local_path(
                'entity_names_%s_0.json' % entity)
            with open(entity_path, 'r') as f:
                names = json.load(f)

            emb_path = self.get_local_output(
                'checkpoints/embeddings_%s_0.v%d.h5' % (entity, self.epochs)
            )
            with h5py.File(emb_path, 'r') as hf:
                embeddings = hf['embeddings'][...]

            col_name = 'reviewerID' if entity == 'user' else entity
            emb_df = pd.DataFrame({
                'embedding': list(embeddings),
                col_name: names
            })

            self.save_parquet(
                dd.from_pandas(emb_df, npartitions=200),
                output_name
            )


class PBGReco(Mario, luigi.Task):
    epochs = luigi.IntParameter()
    dim = luigi.IntParameter()
    loss_fn = luigi.Parameter()
    # hardcoded for now
    # comparator = luigi.Parameter()
    comparator = 'l2'
    lr = luigi.FloatParameter()
    eval_fraction = luigi.FloatParameter()
    regularization_coef = luigi.FloatParameter()
    num_negs = luigi.IntParameter()
    days = luigi.IntParameter()
    min_user_rev = luigi.IntParameter()
    k = luigi.IntParameter()
    item_days = luigi.IntParameter()

    def output_dir(self):
        return 'graph/pbg_reco/days_{DAYS}_min_rev_{MIN_REV}/' \
            '{DIM}_{LOSS_FN}_{COMPARATOR}_{LR}_{REG_COEF}_{NUM_NEGS}_' \
            '{EVAL_FR}_{EPOCHS}/k={K}'.format(
                DIM=self.dim,
                LOSS_FN=self.loss_fn,
                COMPARATOR=self.comparator,
                LR=self.lr,
                REG_COEF=self.regularization_coef,
                EVAL_FR=self.eval_fraction,
                EPOCHS=self.epochs,
                NUM_NEGS=self.num_negs,
                DAYS=self.days,
                MIN_REV=self.min_user_rev,
                K=self.k
            )

    def requires(self):
        return [
            TrainPBG(
                epochs=self.epochs,
                dim=self.dim,
                loss_fn=self.loss_fn,
                comparator=self.comparator,
                lr=self.lr,
                eval_fraction=self.eval_fraction,
                regularization_coef=self.regularization_coef,
                num_negs=self.num_negs,
                days=self.days,
                min_user_rev=self.min_user_rev
            ),
            FilteredDevSet(),
            ItemPopularity(days=self.item_days),
            MostPopularInCatReco(days=self.item_days, k=self.k)
        ]

    def _run(self):
        emb_job, users_job, items_job, fallback_job = self.requires()
        test_users = users_job.load_parquet()[['reviewerID']].compute()
        relevant_items = items_job.load_parquet()[['item']].compute()
        item_embs_df = (
            emb_job.load_parquet('items')
            .merge(relevant_items, on='item')
            .compute()
        )
        user_embs_df = (
            emb_job.load_parquet('users')
            .merge(test_users, on='reviewerID')
            .compute()
        )
        n_users = len(user_embs_df)

        print('now starting nearest neighbour stuff')
        index = faiss.IndexFlatL2(self.dim)
        index.add(np.vstack(item_embs_df.embedding))

        from time import time
        print('now searching')
        start = time()
        D, I = index.search(np.vstack(user_embs_df.embedding), self.k)
        end = time()
        print('done searching !!!')
        print('took %.2f seconds' % (end - start))

        result = pd.DataFrame({
            'rank': np.tile(np.arange(self.k) + 1, n_users),
            'item_ind': I.ravel(),
            'reviewerID': np.repeat(user_embs_df.reviewerID, self.k)
        })

        item_embs_df['item_ind'] = item_embs_df.index
        ind2item = item_embs_df[['item', 'item_ind']]

        pbg_reco = dd.from_pandas(
            result.merge(ind2item, on='item_ind')
            [['reviewerID', 'item', 'rank']],
            chunksize=50000
        )
        fallback = (
            fallback_job.load_parquet()
            [['reviewerID', 'item', 'rank']]
            .rename(columns={'item': 'fallback_item'})
        )

        final = (
            fallback
            .merge(pbg_reco, on=['reviewerID', 'rank'], how='left')
        )
        final['item'] = final['item'].fillna(final['fallback_item'])

        self.save_parquet(final.repartition(npartitions=200))


class MetadataEdges(Mario, luigi.Task):
    """
    brand_edges
    item: string
    brand: string
    brand_encoded: int32 not null

    brand_counts
    brand: string
    count: int64 not null


    cat_edges
    item: string
    category: string
    category_encoded: int32 not null

    cat_counts
    category: string
    count: int64 not null

    feat_edges
    feature: string
    feature_encoded: int32 not null

    feat_counts
    feature: string
    count: int64 not null
    """
    days = luigi.IntParameter()

    def output_dir(self):
        return 'graph/meta_edges/days_%s' % self.days

    def requires(self):
        return OnlyShortMeta(), n_days_subset(self.days)

    def _run(self):
        sc, sqlc = start_spark()
        meta_job, reviews_job = self.requires()
        reviews = reviews_job.load_parquet(sqlc=sqlc)
        relevant_items = reviews.select('item').distinct()

        meta = (
            meta_job
            .load_parquet(sqlc=sqlc)
            .join(relevant_items, on='item')
            .cache()
        )

        brand_edges = (
            meta
            .filter('brand is not null and brand <> ""')
            .selectExpr(
                'item',
                'brand',
                'hash(brand) as brand_encoded'
            )
        )
        self.save_parquet(brand_edges, 'brand_edges')
        self.save_parquet(brand_edges.groupBy('brand').count(), 'brand_counts')

        cat_edges = (
            meta
            .select(
                'item',
                explode('category_fixed').alias('category')
            )
            .selectExpr(
                'item',
                'category',
                'hash(category) as category_encoded'
            )
        )
        self.save_parquet(cat_edges, 'cat_edges')
        self.save_parquet(cat_edges.groupBy('category').count(), 'cat_counts')

        feat_edges = (
            meta
            .select(
                'item',
                explode('feature_fixed').alias('feature')
            )
            .selectExpr(
                'item',
                'feature',
                'hash(feature) as feature_encoded'
            )
        )
        self.save_parquet(feat_edges, 'feat_edges')
        self.save_parquet(feat_edges.groupBy('feature').count(), 'feat_counts')


class MakeEdgeListV2(Mario, luigi.Task):
    """
    saves edges.tsv - edge list for the interactions graph
    and a dataframe of all users under
    'users'
    """
    days = luigi.IntParameter()
    min_user_rev = luigi.IntParameter()
    min_meta_count = luigi.IntParameter()

    def output_dir(self):
        return 'graph/edge_list_v2/days_%s_min_rev_%s_min_meta_%s' % (
            self.days, self.min_user_rev, self.min_meta_count)

    def requires(self):
        return n_days_subset(self.days), MetadataEdges(days=self.days)

    def _run(self):
        reviews_job, meta_job = self.requires()
        reviews = reviews_job.load_parquet()[
            ['reviewerID', 'item']]
        review_counts = reviews.groupby('reviewerID').count().reset_index()
        users_subset = review_counts[review_counts.item >= self.min_user_rev][
            ['reviewerID']]

        # self.save_parquet(users_subset, 'users')

        reviews1 = reviews.merge(users_subset, on='reviewerID').compute()
        print('\n\nfound %s edges matching criteria\n\n' % len(reviews1))

        graph_path = self.local_path('edges.tsv')
        with open(graph_path, 'w') as f:
            for reviewer, item in zip(reviews1.reviewerID, reviews1.item):
                f.write(reviewer + '\treviewed\t' + item + '\n')

            del reviews1

            brands = meta_job.load_parquet('brand_counts')
            good_brands = brands[brands['count'] >= self.min_meta_count]

            brand_edges = (
                meta_job.load_parquet('brand_edges')
                .merge(good_brands, on='brand')
                [['item', 'brand_encoded']]
                .compute()
            )
            i = 100
            for item, brand in zip(
                    brand_edges.item, brand_edges.brand_encoded):
                i -= 1
                if i == 0:
                    break
                f.write(item + '\thas_brand\t' + str(brand) + '\n')

            cats = meta_job.load_parquet('cat_counts')
            good_cats = cats[cats['count'] >= self.min_meta_count]

            cat_edges = (
                meta_job.load_parquet('cat_edges')
                .merge(good_cats, on='category')
                [['item', 'category_encoded']]
                .compute()
            )
            i = 100
            for item, cat in zip(cat_edges.item, cat_edges.category_encoded):
                i -= 1
                if i == 0:
                    break
                f.write(item + '\thas_category\t' + str(cat) + '\n')

            feats = meta_job.load_parquet('feat_counts')
            good_feats = feats[feats['count'] >= self.min_meta_count]

            feat_edges = (
                meta_job.load_parquet('feat_edges')
                .merge(good_feats, on='feature')
                [['item', 'feature_encoded']]
                .compute()
            )
            i = 100
            for item, feat in zip(feat_edges.item, feat_edges.feature_encoded):
                i -= 1
                if i == 0:
                    break
                f.write(item + '\thas_feature\t' + str(feat) + '\n')

        self.backup_local_dir()


class PreparePBGInputV2(PreparePBGInput):
    days = luigi.IntParameter()
    min_user_rev = luigi.IntParameter()
    min_meta_count = luigi.IntParameter()

    def pbg_config(self):
        config = super().pbg_config()
        config.update(dict(
            entities={
                "user": {"num_partitions": 1},
                "item": {"num_partitions": 1},
                "brand": {"num_partitions": 1},
                "category": {"num_partitions": 1},
                "feature": {"num_partitions": 1}
            },
            relations=[
                {
                    "name": "reviewed",
                    "lhs": "user",
                    "rhs": "item",
                    "operator": "none",
                },
                {
                    "name": "has_brand",
                    "lhs": "item",
                    "rhs": "brand",
                    "operator": "diagonal",
                },
                {
                    "name": "has_category",
                    "lhs": "item",
                    "rhs": "category",
                    "operator": "diagonal",
                },
                {
                    "name": "has_feature",
                    "lhs": "item",
                    "rhs": "feature",
                    "operator": "diagonal",
                },
            ],
        ))
        return config

    def output_dir(self):
        return 'graph/pbg_input/days_%s_min_rev_%s_min_meta_%s' % (
            self.days, self.min_user_rev, self.min_meta_count)

    def requires(self):
        return MakeEdgeListV2(
            days=self.days,
            min_user_rev=self.min_user_rev,
            min_meta_count=self.min_meta_count)


class TrainPBGV2(TrainPBG):
    """Trains embeddings. Saves

    users
    embedding: list<item: float>
    child 0, item: float
    reviewerID: string

    items
    embedding: list<item: float>
        child 0, item: float
    item: string

    brands
    embedding: list<item: float>
        child 0, item: float
    brand: string

    categories
    embedding: list<item: float>
        child 0, item: float
    category: string

    features
    embedding: list<item: float>
        child 0, item: float
    feature: string
    """
    min_meta_count = luigi.IntParameter()

    def output_dir(self):
        return 'graph/train_pbg_v2/days_{DAYS}_min_rev_{MIN_REV}_min_meta_{MIN_META}/' \
            '{DIM}_{LOSS_FN}_{COMPARATOR}_{LR}_{REG_COEF}_{NUM_NEGS}_' \
            '{EVAL_FR}_{EPOCHS}'.format(
                DIM=self.dim,
                LOSS_FN=self.loss_fn,
                COMPARATOR=self.comparator,
                LR=self.lr,
                REG_COEF=self.regularization_coef,
                EVAL_FR=self.eval_fraction,
                EPOCHS=self.epochs,
                NUM_NEGS=self.num_negs,
                DAYS=self.days,
                MIN_REV=self.min_user_rev,
                MIN_META=self.min_meta_count
            )

    def requires(self):
        return PreparePBGInputV2(
            days=self.days,
            min_user_rev=self.min_user_rev,
            min_meta_count=self.min_meta_count)

    # entity name, subdirectory name, column name
    entities = [
        ('user', 'users', 'reviewerID'),
        ('item', 'items', 'item'),
        ('brand', 'brands', 'brand'),
        ('category', 'categories', 'category'),
        ('feature', 'features', 'feature')
    ]


class PBGRecoV2(PBGReco):
    min_meta_count = luigi.IntParameter()

    def output_dir(self):
        return 'graph/pbg_reco_v2/days_{DAYS}_min_rev_{MIN_REV}_min_meta_{MIN_META}/' \
            '{DIM}_{LOSS_FN}_{COMPARATOR}_{LR}_{REG_COEF}_{NUM_NEGS}_' \
            '{EVAL_FR}_{EPOCHS}/k={K}'.format(
                DIM=self.dim,
                LOSS_FN=self.loss_fn,
                COMPARATOR=self.comparator,
                LR=self.lr,
                REG_COEF=self.regularization_coef,
                EVAL_FR=self.eval_fraction,
                EPOCHS=self.epochs,
                NUM_NEGS=self.num_negs,
                DAYS=self.days,
                MIN_REV=self.min_user_rev,
                MIN_META=self.min_meta_count,
                K=self.k
            )

    def requires(self):
        return [
            TrainPBGV2(
                epochs=self.epochs,
                dim=self.dim,
                loss_fn=self.loss_fn,
                comparator=self.comparator,
                lr=self.lr,
                eval_fraction=self.eval_fraction,
                regularization_coef=self.regularization_coef,
                num_negs=self.num_negs,
                days=self.days,
                min_user_rev=self.min_user_rev,
                min_meta_count=self.min_meta_count
            ),
            FilteredDevSet(),
            ItemPopularity(days=self.item_days),
            MostPopularInCatReco(days=self.item_days, k=self.k)
        ]


class Sink(Mario, luigi.Task):
    def output_dir(self):
        return 'graph/sink'

    def requires(self):
        return [
            MakeEdgeListV2(
                days=2,
                min_user_rev=2,
                min_meta_count=20
            ),
            TrainPBGV2(
                days=2,
                min_user_rev=2,
                min_meta_count=20,
                num_negs=100,
                epochs=2,
                eval_fraction=0.05,
                regularization_coef=0.001,
                lr=0.1,
                comparator='l2',
                dim=4,
                loss_fn='softmax'
            ),
            TrainPBG(
                days=2,
                min_user_rev=2,
                num_negs=100,
                epochs=2,
                eval_fraction=0.05,
                regularization_coef=0.001,
                lr=0.1,
                comparator='l2',
                dim=4,
                loss_fn='softmax'
            ),
            TrainPBGV2(
                days=2,
                min_user_rev=2,
                min_meta_count=20,
                num_negs=100,
                epochs=3,
                eval_fraction=0.05,
                regularization_coef=0.001,
                lr=0.1,
                comparator='l2',
                dim=4,
                loss_fn='softmax'
            ),
            PBGRecoV2(
                days=2,
                min_user_rev=2,
                min_meta_count=20,
                num_negs=100,
                epochs=3,
                eval_fraction=0.05,
                regularization_coef=0.001,
                lr=0.1,
                # comparator='l2',
                dim=4,
                loss_fn='softmax',
                k=10,
                item_days=2
            )
        ]

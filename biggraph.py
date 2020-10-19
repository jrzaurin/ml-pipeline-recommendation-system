from pathlib import Path
import luigi
import dask.dataframe as dd
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging

from utils import Mario
from x04_train_test_split import n_days_subset, DEV_SET_REVIEWS_JOB, TRAIN_SET_REVIEWS_JOB, ONE_YEAR_REVIEWS_JOB, MINI_REVIEWS_JOB
from baselines import MostPopularReco, RandomReco, MostPopularInCatReco


class MakeEdgeList(Mario, luigi.Task):
    days = luigi.IntParameter()

    def output_dir(self):
        return 'graph/edge_list/days_%s' % self.days

    def requires(self):
        return n_days_subset(self.days)

    def _run(self):
        reviews = self.requires().load_parquet()[
            ['reviewerID', 'item']].compute()

        self.clean_local_dir()
        output_dir = self.local_path()

        graph_path = self.local_path('edges.tsv')
        with open(graph_path, 'w') as f:
            for reviewer, item in zip(reviews.reviewerID, reviews.item):
                f.write(reviewer + '\t' + item + '\n')

        self.backup_local_dir()


class PreparePBGInput(Mario, luigi.Task):
    days = luigi.IntParameter()

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
                "WHATEVER": {"num_partitions": 1}
            },
            relations=[
                {
                    "name": "doesnt_matter",
                    "lhs": "WHATEVER",
                    "rhs": "WHATEVER",
                    "operator": "complex_diagonal",
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
        return 'graph/pbg_input/days_%s' % self.days

    def requires(self):
        return MakeEdgeList(days=self.days)

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
            TSVEdgelistReader(lhs_col=0, rel_col=None, rhs_col=1),
            dynamic_relations=config.dynamic_relations,
        )

        self.backup_local_dir()


class TrainPBG(Mario, luigi.Task):
    epochs = luigi.IntParameter()
    dim = luigi.IntParameter()
    loss_fn = luigi.Parameter()
    comparator = luigi.Parameter()
    lr = luigi.FloatParameter()
    eval_fraction = luigi.FloatParameter()
    regularization_coef = luigi.FloatParameter()
    num_negs = luigi.IntParameter()
    days = luigi.IntParameter()

    def output_dir(self):
        return 'graph/train_pbg/days_{DAYS}/' \
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
                DAYS=self.days
            )

    def requires(self):
        return PreparePBGInput(days=self.days)

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

    def _run(self):
        setup_logging()
        config = parse_config(self.pbg_config())
        subprocess_init = SubprocessInitializer()
        train(config, subprocess_init=subprocess_init)
        self.backup_local_dir()


class Sink(Mario, luigi.Task):
    def output_dir(self):
        return 'graph/sink'

    def requires(self):
        return [
            TrainPBG(
                epochs=2,
                dim=100,
                loss_fn='softmax',
                comparator='dot',
                lr=0.1,
                eval_fraction=0.05,
                regularization_coef=1e-3,
                num_negs=1000,
                days=2
            ),

            TrainPBG(
                epochs=200,
                dim=100,
                loss_fn='softmax',
                comparator='l2',
                lr=0.1,
                eval_fraction=0.05,
                regularization_coef=1e-3,
                num_negs=1000,
                days=365
            ),

            TrainPBG(
                epochs=200,
                dim=200,
                loss_fn='softmax',
                comparator='l2',
                lr=0.1,
                eval_fraction=0.05,
                regularization_coef=1e-3,
                num_negs=1000,
                days=365
            )
        ]

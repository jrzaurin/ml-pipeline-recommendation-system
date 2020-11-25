import argparse


def gmf_parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir", type=str, default="data/processed/amazon", help="data directory."
    )
    parser.add_argument(
        "--dataname",
        type=str,
        default="leave_one_out_w_negative_full_valid.npz",
        help="npz file with dataset",
    )
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size.")
    parser.add_argument("--n_emb", type=int, default=8, help="embedding size.")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="if lr_scheduler this will be max_lr"
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="adamw",
        help="Specify an optimizer: adamw or sgd",
    )
    parser.add_argument(
        "--lr_scheduler",
        action="store_true",
        help="if true use ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1,
        help="Patience for ReduceLROnPlateau lr_scheduler before decreasing lr, \
        By default we eval_every 2 epochs",
    )
    parser.add_argument(
        "--eval_every", type=int, default=2, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--n_neg",
        type=int,
        default=4,
        help="number of negative instances to consider per positive instance",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="number of items to retrieve for recommendation",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Patience for early stopping. By default we eval_every 2 epochs",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    return parser.parse_args()


def mlp_parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir", type=str, default="data/processed/amazon", help="data directory."
    )
    parser.add_argument(
        "--dataname",
        type=str,
        default="leave_one_out_w_negative_full_valid.npz",
        help="npz file with dataset",
    )
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size.")
    parser.add_argument(
        "--layers",
        type=str,
        default="[32, 16, 8]",
        help="layer architecture. The first elements is used for the embedding \
        layers and equals n_emb*2",
    )
    parser.add_argument(
        "--dropouts",
        type=str,
        default="[0., 0.]",
        help="dropout per dense layer. len(dropouts) = len(layers)-1",
    )
    parser.add_argument("--l2reg", type=float, default=0.0, help="l2 regularization")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="if lr_scheduler this will be max_lr"
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="adamw",
        help="Specify an optimizer: adamw or sgd",
    )
    parser.add_argument(
        "--lr_scheduler",
        action="store_true",
        help="if true use ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1,
        help="Patience for ReduceLROnPlateau lr_scheduler before decreasing lr, \
        By default we eval_every 2 epochs",
    )
    parser.add_argument(
        "--eval_every", type=int, default=2, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--n_neg",
        type=int,
        default=4,
        help="number of negative instances to consider per positive instance",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="number of items to retrieve for recommendation",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Patience for early stopping. By default we eval_every 2 epochs",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    return parser.parse_args()


def ncf_parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir", type=str, default="data/processed/amazon", help="data directory."
    )
    parser.add_argument(
        "--dataname",
        type=str,
        default="leave_one_out_w_negative_full_valid.npz",
        help="npz file with dataset",
    )

    # GMF set up
    parser.add_argument("--n_emb", type=int, default=8, help="embedding size.")

    # MLP set up
    parser.add_argument(
        "--layers",
        type=str,
        default="[32, 16, 8]",
        help="layer architecture. The first elements is used for the embedding \
        layers for the MLP part and equals n_emb*2",
    )
    parser.add_argument(
        "--dropouts",
        type=str,
        default="[0., 0.]",
        help="dropout per dense layer. len(dropouts) = len(layers)-1",
    )

    # train/eval parameter
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate.")
    parser.add_argument(
        "--learner",
        type=str,
        default="adamw",
        help="Specify an optimizer: adamw or sgd",
    )
    parser.add_argument("--l2reg", type=float, default=0.0, help="l2 regularization.")
    parser.add_argument(
        "--lr_scheduler",
        action="store_true",
        help="if true use ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1,
        help="Patience for ReduceLROnPlateau lr_scheduler before decreasing lr, \
        By default we eval_every 2 epochs so lr_patience of 2 implies 4 epochs",
    )
    parser.add_argument(
        "--eval_every", type=int, default=2, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--n_neg",
        type=int,
        default=4,
        help="number of negative instances to consider per positive instance",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="number of items to retrieve for recommendation",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Patience for early stopping. By default we eval_every 2 epochs",
    )

    # Pretrained model names
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="freeze all but the last output layer where \
        weights are combined",
    )
    parser.add_argument(
        "--mf_pretrain",
        type=str,
        default="",
        help="Specify the pretrain model filename for GMF part. \
        If empty, no pretrain will be used",
    )
    parser.add_argument(
        "--mlp_pretrain",
        type=str,
        default="",
        help="Specify the pretrain model filename for MLP part. \
        If empty, no pretrain will be used",
    )

    # save
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    return parser.parse_args()

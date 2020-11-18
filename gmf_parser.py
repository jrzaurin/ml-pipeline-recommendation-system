import argparse


def parse_args():

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
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size.")
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
        default=4,
        help="Patience for ReduceLROnPlateau lr_scheduler before decreasing lr",
    )
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
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
        default=5,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    return parser.parse_args()

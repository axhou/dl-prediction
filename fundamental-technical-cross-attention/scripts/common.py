import argparse

from src.config import get_preset


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-rows", type=int, default=0, help="Use 0 for full data.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    return parser


def configured_preset(name, seed, args, **overrides):
    shared = {"random_state": seed, **overrides}
    if args.epochs is not None:
        shared["epochs"] = args.epochs
    if args.batch_size is not None:
        shared["batch_size"] = args.batch_size
    if args.patience is not None:
        shared["patience"] = args.patience
    return get_preset(name, **shared)

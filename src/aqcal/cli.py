"""Command-line interface: ``python -m aqcal run --data FILE --out DIR``."""

from __future__ import annotations

import argparse
import sys
import time

from aqcal.experiment import RunConfig, run_experiment
from aqcal.report import format_summary, write_outputs


def _fraction(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _fold_count(text: str) -> int:
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError("needs at least 2 folds")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aqcal",
        description="Compare CO calibration models for a low-cost metal-oxide sensor array.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    defaults = RunConfig()
    run = sub.add_parser("run", help="train, tune and score all models, then write results")
    run.add_argument("--data", required=True, help="path to AirQualityUCI.csv")
    run.add_argument("--out", required=True, help="directory for metrics.json, metrics.csv, feature_importance.csv")
    run.add_argument("--train-frac", type=_fraction, default=defaults.train_frac,
                     help="share of rows (earliest) used for training (default: %(default)s)")
    run.add_argument("--val-frac", type=_fraction, default=defaults.val_frac,
                     help="share of rows after the training block used for model selection (default: %(default)s)")
    run.add_argument("--cv-splits", type=_fold_count, default=defaults.cv_splits,
                     help="TimeSeriesSplit folds for tuning, also the stacking folds (default: %(default)s)")
    run.add_argument("--search-iterations", type=_positive_int, default=defaults.search_iterations,
                     help="RandomizedSearchCV candidates for the boosting model (default: %(default)s)")
    run.add_argument("--forest-trees", type=_positive_int, default=defaults.forest_trees,
                     help="trees in the random forest (default: %(default)s)")
    run.add_argument("--importance-repeats", type=_positive_int, default=defaults.importance_repeats,
                     help="shuffles per feature for permutation importance (default: %(default)s)")
    run.add_argument("--seed", type=int, default=defaults.seed, help="random_state for all models (default: %(default)s)")
    run.add_argument("--n-jobs", type=int, default=defaults.n_jobs,
                     help="parallel jobs for the two searches and the random forest; -1 uses all cores (default: %(default)s)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.train_frac + args.val_frac >= 1.0:
        print("error: --train-frac plus --val-frac must be below 1", file=sys.stderr)
        return 2

    config = RunConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        cv_splits=args.cv_splits,
        search_iterations=args.search_iterations,
        forest_trees=args.forest_trees,
        importance_repeats=args.importance_repeats,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    started = time.perf_counter()
    try:
        result = run_experiment(args.data, config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    paths = write_outputs(result, args.out)

    print(format_summary(result))
    print()
    for path in paths:
        print(f"wrote {path}")
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    return 0

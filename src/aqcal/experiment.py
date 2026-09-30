"""End-to-end comparison of calibration models on the hourly record.

Protocol
--------
1. Parse the export and keep rows with a target and at least one sensor value.
2. Split in time order into train / validation / test blocks.
3. Tune the ridge penalty and the boosting hyper-parameters with
   ``TimeSeriesSplit`` folds drawn from the training block only.
4. Fit every model on the training block and score it on the validation
   block; the model with the lowest validation RMSE is the selected model.
5. Refit every model on training + validation and score it once on the test
   block, which no step above has seen.
6. Compute permutation importance of the selected model on the test block.
"""

from __future__ import annotations

import hashlib
import os
import platform
from dataclasses import asdict, dataclass, field
from typing import Callable

import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from aqcal import models
from aqcal.dataset import load_air_quality
from aqcal.features import FEATURE_COLUMNS, TARGET, check_no_leakage, prepare_dataset
from aqcal.metrics import score_all
from aqcal.splits import chronological_split

MODEL_ORDER = (
    "mean_baseline",
    "single_sensor_linear",
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
    "stacking",
)


@dataclass(frozen=True)
class RunConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    cv_splits: int = 5
    search_iterations: int = 20
    forest_trees: int = 300
    importance_repeats: int = 10
    seed: int = 42
    n_jobs: int | None = -1


@dataclass
class ExperimentResult:
    dataset: dict
    periods: dict
    tuning: dict
    scores: dict
    selected_model: str
    importance: pd.DataFrame
    config: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def scores_table(self) -> pd.DataFrame:
        rows = []
        for name in MODEL_ORDER:
            entry = self.scores[name]
            row = {"model": name}
            for stage in ("validation", "test"):
                for metric, value in entry[stage].items():
                    row[f"{stage}_{metric}"] = value
            rows.append(row)
        return pd.DataFrame(rows)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_model(scores: dict[str, dict], order: tuple[str, ...] = MODEL_ORDER) -> str:
    """Return the name of the model with the lowest validation RMSE.

    ``scores`` maps model names to ``{"validation": {...}, "test": {...}}``.
    Only the validation entry is read, so test scores cannot influence the
    choice. Models are considered in ``order`` and a tie goes to the one listed
    first; names missing from ``scores`` are skipped.
    """
    candidates = [name for name in order if name in scores]
    if not candidates:
        raise ValueError("no scored models to choose from")
    if any(not np.isfinite(scores[name]["validation"]["rmse"]) for name in candidates):
        raise ValueError("validation RMSE must be finite for every candidate")
    return min(candidates, key=lambda name: scores[name]["validation"]["rmse"])


def build_factories(config: RunConfig, ridge_alpha: float, hgb_params: dict) -> dict[str, Callable[[], BaseEstimator]]:
    """Map each model name to a zero-argument function returning a fresh, unfitted estimator."""
    seed, trees, jobs = config.seed, config.forest_trees, config.n_jobs
    return {
        "mean_baseline": models.mean_baseline,
        "single_sensor_linear": models.single_sensor_linear,
        "ridge": lambda: models.ridge(ridge_alpha),
        "random_forest": lambda: models.random_forest(trees, seed, jobs),
        "hist_gradient_boosting": lambda: models.hist_gradient_boosting(seed, **hgb_params),
        "stacking": lambda: models.stacking(
            trees, seed, jobs, folds=config.cv_splits
        ),
    }


def run_experiment(data_path: str, config: RunConfig | None = None) -> ExperimentResult:
    config = config or RunConfig()
    frame = load_air_quality(data_path)
    prepared = prepare_dataset(frame)
    check_no_leakage(prepared.X.columns)
    split = chronological_split(prepared.X, prepared.y, config.train_frac, config.val_frac)

    folds = TimeSeriesSplit(n_splits=config.cv_splits)
    ridge_search = models.tune_ridge(split.X_train, split.y_train, folds, config.n_jobs)
    hgb_search = models.tune_hist_gradient_boosting(
        split.X_train,
        split.y_train,
        folds,
        n_iter=config.search_iterations,
        seed=config.seed,
        n_jobs=config.n_jobs,
    )

    factories = build_factories(config, ridge_search["alpha"], hgb_search["params"])
    X_fit_final, y_fit_final = split.X_train_val, split.y_train_val

    scores: dict[str, dict] = {}
    stacking_protocol = {
        "base_parameter_policy": "fixed before fitting; independent of standalone tuning",
        "ridge_alpha": models.STACK_RIDGE_ALPHA,
        "hist_gradient_boosting": dict(models.STACK_HGB_PARAMS),
        "meta_model": "LinearRegression(positive=True), intercept enabled",
        "oof_policy": "expanding-window past-only; initial warmup excluded from meta fit",
    }
    for name in MODEL_ORDER:
        make = factories[name]
        on_train = make().fit(split.X_train, split.y_train)
        validation = score_all(split.y_val, on_train.predict(split.X_val))

        scores[name] = {"validation": validation}
        if name == "stacking":
            stacking_protocol["validation_fit"] = {
                "warmup_rows": on_train.warmup_rows_,
                "meta_rows": len(on_train.oof_predictions_),
                "folds": on_train.fold_records_,
            }
    # Select before any model is refitted or any held-out target is scored.
    selected = select_model(scores)

    final_models: dict[str, BaseEstimator] = {}
    for name in MODEL_ORDER:
        final = factories[name]().fit(X_fit_final, y_fit_final)
        scores[name]["test"] = score_all(split.y_test, final.predict(split.X_test))
        final_models[name] = final
        if name == "stacking":
            stacking_protocol["test_fit"] = {
                "warmup_rows": final.warmup_rows_,
                "meta_rows": len(final.oof_predictions_),
                "folds": final.fold_records_,
            }

    perm = permutation_importance(
        final_models[selected],
        split.X_test,
        split.y_test,
        scoring=models.SCORING,
        n_repeats=config.importance_repeats,
        random_state=config.seed,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": list(split.X_test.columns),
                "rmse_increase_mean": perm.importances_mean,
                "rmse_increase_std": perm.importances_std,
            }
        )
        .sort_values("rmse_increase_mean", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

    dataset = {
        "file": os.path.basename(str(data_path)),
        "sha256": _sha256(str(data_path)),
        "target": TARGET,
        "features": list(FEATURE_COLUMNS),
        "rows_parsed": prepared.rows_parsed,
        "rows_missing_target": prepared.rows_missing_target,
        "rows_without_sensor_data": prepared.rows_without_sensor_data,
        "rows_used": prepared.rows_used,
    }
    tuning = {
        "cv": f"TimeSeriesSplit(n_splits={config.cv_splits}) on the training block",
        "ridge": ridge_search,
        "hist_gradient_boosting": hgb_search,
        "stacking": stacking_protocol,
    }
    return ExperimentResult(
        dataset=dataset,
        periods=split.periods(),
        tuning=tuning,
        scores=scores,
        selected_model=selected,
        importance=importance,
        config=asdict(config),
        environment={
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
        },
    )

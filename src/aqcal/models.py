"""Estimator factories and time-series-aware hyper-parameter search.

Every estimator is a scikit-learn ``Pipeline`` that takes the feature
DataFrame produced by :func:`aqcal.features.prepare_dataset`. Imputation and
scaling live inside the pipelines, so their statistics are learned from
whatever rows the pipeline is fitted on and never from later periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aqcal.features import CO_SENSOR, CONTINUOUS_COLUMNS, FEATURE_COLUMNS, TIME_COLUMNS

SCORING = "neg_root_mean_squared_error"

RIDGE_ALPHAS = np.logspace(-3, 3, 13)

HGB_SEARCH_SPACE = {
    "model__learning_rate": loguniform(0.02, 0.3),
    "model__max_iter": randint(100, 401),
    "model__max_leaf_nodes": randint(8, 64),
    "model__min_samples_leaf": randint(10, 101),
    "model__l2_regularization": loguniform(1e-4, 10.0),
}


def mean_baseline() -> DummyRegressor:
    """Predicts the mean of the training target for every hour."""
    return DummyRegressor(strategy="mean")


def single_sensor_linear() -> Pipeline:
    """Ordinary least squares on the CO-targeted sensor only.

    This is the classic one-sensor linear calibration; every other column is
    dropped by the column selector.
    """
    select = ColumnTransformer(
        [("co_sensor", SimpleImputer(strategy="median"), [CO_SENSOR])],
        remainder="drop",
    )
    return Pipeline([("select", select), ("model", LinearRegression())])


def _linear_preprocessor() -> ColumnTransformer:
    continuous = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    calendar = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [
            ("continuous", continuous, list(CONTINUOUS_COLUMNS)),
            ("calendar", calendar, list(TIME_COLUMNS)),
        ],
        remainder="drop",
    )


def ridge(alpha: float = 1.0) -> Pipeline:
    """Ridge regression on all features (hour and weekday one-hot encoded)."""
    return Pipeline([("prep", _linear_preprocessor()), ("model", Ridge(alpha=alpha))])


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("all", SimpleImputer(strategy="median"), list(FEATURE_COLUMNS))],
        remainder="drop",
    )


def random_forest(n_estimators: int = 300, seed: int = 42, n_jobs: int | None = None) -> Pipeline:
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return Pipeline([("prep", _tree_preprocessor()), ("model", forest)])


def hist_gradient_boosting(seed: int = 42, **params) -> Pipeline:
    """Histogram gradient boosting; early stopping is off so ``max_iter`` is honoured."""
    booster = HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **params)
    return Pipeline([("prep", _tree_preprocessor()), ("model", booster)])


def stacking(
    ridge_alpha: float,
    hgb_params: dict,
    n_estimators: int = 300,
    seed: int = 42,
    n_jobs: int | None = None,
    folds: int = 5,
) -> StackingRegressor:
    """Blend ridge, random forest and boosting with non-negative linear weights.

    The meta-model is trained on out-of-fold predictions from contiguous,
    unshuffled folds of the rows passed to ``fit``.
    """
    base = [
        ("ridge", ridge(ridge_alpha)),
        ("random_forest", random_forest(n_estimators, seed, n_jobs)),
        ("hist_gradient_boosting", hist_gradient_boosting(seed, **hgb_params)),
    ]
    return StackingRegressor(
        estimators=base,
        final_estimator=LinearRegression(positive=True),
        cv=KFold(n_splits=folds, shuffle=False),
    )


def _plain(value):
    """Convert numpy scalars to Python numbers so results serialise to JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def tune_ridge(
    X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit, n_jobs: int | None = None
) -> dict:
    """Grid search over the ridge penalty using forward-chaining folds."""
    search = GridSearchCV(
        ridge(),
        {"model__alpha": RIDGE_ALPHAS},
        scoring=SCORING,
        cv=cv,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return {
        "alpha": float(search.best_params_["model__alpha"]),
        "cv_rmse": float(-search.best_score_),
        "candidates": int(len(RIDGE_ALPHAS)),
    }


def tune_hist_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    n_iter: int = 20,
    seed: int = 42,
    n_jobs: int | None = None,
) -> dict:
    """Randomised search for the boosting model using forward-chaining folds."""
    search = RandomizedSearchCV(
        hist_gradient_boosting(seed),
        HGB_SEARCH_SPACE,
        n_iter=n_iter,
        scoring=SCORING,
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    params = {
        key.split("__", 1)[1]: _plain(value) for key, value in sorted(search.best_params_.items())
    }
    return {"params": params, "cv_rmse": float(-search.best_score_), "candidates": int(n_iter)}


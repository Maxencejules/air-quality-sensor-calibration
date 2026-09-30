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
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from aqcal.features import CO_SENSOR, CONTINUOUS_COLUMNS, FEATURE_COLUMNS, TIME_COLUMNS, check_no_leakage

SCORING = "neg_root_mean_squared_error"

RIDGE_ALPHAS = np.logspace(-3, 3, 13)

# Prespecified, not selected using any target values. The separately tuned
# standalone models must not inject later-fold target information into OOF.
STACK_RIDGE_ALPHA = 1.0
STACK_HGB_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}

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
        [("co_sensor", SimpleImputer(strategy="median", keep_empty_features=True), [CO_SENSOR])],
        remainder="drop",
    )
    return Pipeline([("select", select), ("model", LinearRegression())])


def _linear_preprocessor() -> ColumnTransformer:
    continuous = Pipeline(
        [("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]
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
        [("all", SimpleImputer(strategy="median", keep_empty_features=True), list(FEATURE_COLUMNS))],
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


class ForwardStackingRegressor(RegressorMixin, BaseEstimator):
    """Stack using predictions trained strictly before their target timestamps.

    Expanding-window folds leave an initial warmup block without predictions;
    only covered rows train the meta-model. Afterwards each base pipeline is
    refitted on all input rows for inference. Base parameters are fixed before
    ``fit``; no tuning on later fold targets is performed here.
    """

    def __init__(self, estimators, folds=5, final_estimator=None):
        self.estimators = estimators
        self.folds = folds
        self.final_estimator = final_estimator

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("forward stacking requires a timestamped DataFrame and Series")
        if not isinstance(X.index, pd.DatetimeIndex) or X.index.hasnans:
            raise ValueError("forward stacking requires valid timestamps")
        if not X.index.equals(y.index):
            raise ValueError("feature and target timestamps must match")
        if not X.index.is_monotonic_increasing or X.index.has_duplicates:
            raise ValueError("forward stacking requires strictly increasing timestamps")
        if isinstance(self.folds, bool) or not isinstance(self.folds, int) or self.folds < 2:
            raise ValueError("folds must be an integer of at least 2")
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError("target values must be finite")
        check_no_leakage(X.columns)
        if not self.estimators or len({name for name, _ in self.estimators}) != len(self.estimators):
            raise ValueError("base estimators must have distinct names")

        predictions = np.full((len(X), len(self.estimators)), np.nan)
        records = []
        for train, holdout in TimeSeriesSplit(n_splits=self.folds).split(X):
            for column, (_, template) in enumerate(self.estimators):
                fitted = clone(template).fit(X.iloc[train], y.iloc[train])
                values = np.asarray(fitted.predict(X.iloc[holdout]), dtype=float).reshape(-1)
                if len(values) != len(holdout) or not np.isfinite(values).all():
                    raise ValueError("base estimator returned invalid out-of-fold predictions")
                predictions[holdout, column] = values
            records.append({
                "train_rows": len(train), "oof_rows": len(holdout),
                "train_start": X.index[train[0]].isoformat(),
                "train_end": X.index[train[-1]].isoformat(),
                "oof_start": X.index[holdout[0]].isoformat(),
                "oof_end": X.index[holdout[-1]].isoformat(),
            })
        covered = np.isfinite(predictions).all(axis=1)
        self.oof_predictions_ = pd.DataFrame(
            predictions[covered], index=X.index[covered],
            columns=[name for name, _ in self.estimators],
        )
        meta = self.final_estimator if self.final_estimator is not None else LinearRegression(positive=True)
        self.final_estimator_ = clone(meta).fit(self.oof_predictions_, y.iloc[covered])
        self.estimators_ = [
            (name, clone(template).fit(X, y)) for name, template in self.estimators
        ]
        self.fold_records_ = records
        self.warmup_rows_ = int((~covered).sum())
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame):
        check_is_fitted(self, "estimators_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError("prediction features must match the fitted columns in order")
        predictions = pd.DataFrame(
            {name: estimator.predict(X) for name, estimator in self.estimators_}, index=X.index
        )
        return self.final_estimator_.predict(predictions)


def stacking(
    n_estimators: int = 300,
    seed: int = 42,
    n_jobs: int | None = None,
    folds: int = 5,
) -> ForwardStackingRegressor:
    """Blend ridge, random forest and boosting with non-negative linear weights.

    Standalone tuning results are deliberately not accepted by this factory.
    Each OOF prediction and its fitted preprocessing use past rows only.
    """
    base = [
        ("ridge", ridge(STACK_RIDGE_ALPHA)),
        ("random_forest", random_forest(n_estimators, seed, n_jobs)),
        ("hist_gradient_boosting", hist_gradient_boosting(seed, **STACK_HGB_PARAMS)),
    ]
    return ForwardStackingRegressor(
        estimators=base,
        final_estimator=LinearRegression(positive=True),
        folds=folds,
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


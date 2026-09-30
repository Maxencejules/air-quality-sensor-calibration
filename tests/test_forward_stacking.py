"""Adversarial temporal tests; inspect actual estimator inputs, not fold labels."""

import unittest
from unittest import mock

import support  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from aqcal import models
from aqcal.experiment import RunConfig, build_factories
from aqcal.splits import chronological_split


class RecordingMean(RegressorMixin, BaseEstimator):
    events = []

    def fit(self, X, y):
        self.mean_ = float(y.mean())
        self.end_ = X.index[-1]
        type(self).events.append(("fit", X.copy(), y.copy()))
        return self

    def predict(self, X):
        type(self).events.append(("predict", X.copy(), self.end_))
        return np.full(len(X), self.mean_)


class RecordingMeta(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self

    def predict(self, X):
        return X.mean(axis=1).to_numpy()


class ForwardStackingTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2020-01-01", periods=24, freq="h")
        self.X = pd.DataFrame({"sensor": np.arange(24, dtype=float)}, index=index)
        self.y = pd.Series(np.arange(24, dtype=float), index=index)
        RecordingMean.events = []

    def stack(self):
        # Exercise the actual experiment factory while replacing expensive
        # bases with a transparent hand-calculable estimator.
        return models.stacking(n_estimators=5, folds=3).set_params(
            estimators=[("mean", RecordingMean())], final_estimator=RecordingMeta()
        )

    def test_every_oof_fit_and_prediction_is_strictly_past_only(self):
        fitted = self.stack().fit(self.X, self.y)
        prediction_events = [event for event in RecordingMean.events if event[0] == "predict"]
        self.assertEqual(len(prediction_events), 3)
        for _, holdout, last_training_timestamp in prediction_events:
            self.assertLess(last_training_timestamp, holdout.index.min())
        fits = [event for event in RecordingMean.events if event[0] == "fit"]
        self.assertEqual([len(X) for _, X, _ in fits], [6, 12, 18, 24])
        for _, X, y in fits:
            pd.testing.assert_frame_equal(X, self.X.loc[X.index])
            pd.testing.assert_series_equal(y, self.y.loc[X.index])
        self.assertEqual(len(fitted.fold_records_), 3)

    def test_oof_values_are_hand_computed_prefix_means_and_warmup_is_excluded(self):
        fitted = self.stack().fit(self.X, self.y)
        expected = np.repeat([2.5, 5.5, 8.5], 6)
        np.testing.assert_array_equal(fitted.oof_predictions_["mean"], expected)
        self.assertEqual(fitted.warmup_rows_, 6)
        pd.testing.assert_series_equal(fitted.final_estimator_.y_, self.y.iloc[6:])
        pd.testing.assert_frame_equal(fitted.final_estimator_.X_, fitted.oof_predictions_)

    def test_future_feature_and_target_poison_cannot_change_earlier_oof(self):
        template = Pipeline([("impute", SimpleImputer(keep_empty_features=True)), ("model", Ridge())])
        before = models.ForwardStackingRegressor([("ridge", template)], folds=3).fit(self.X, self.y)
        X, y = self.X.copy(), self.y.copy()
        X.iloc[12:] = 1e6
        y.iloc[12:] = -1e6
        after = models.ForwardStackingRegressor([("ridge", template)], folds=3).fit(X, y)
        # Predictions for hours 6..11 cannot use rows at hour 12 or later.
        np.testing.assert_array_equal(before.oof_predictions_.iloc[:6], after.oof_predictions_.iloc[:6])

    def test_empty_early_feature_has_stable_width_and_no_future_median(self):
        X = self.X.copy()
        X.iloc[:6] = np.nan
        template = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("model", Ridge())])
        fitted = models.ForwardStackingRegressor([("ridge", template)], folds=3).fit(X, self.y)
        np.testing.assert_allclose(fitted.oof_predictions_.iloc[:6, 0], 2.5)

    def test_stacking_parameters_do_not_depend_on_standalone_search_results(self):
        config = RunConfig(forest_trees=5, cv_splits=3, n_jobs=1)
        first = build_factories(config, 0.001, {"max_iter": 150})["stacking"]()
        second = build_factories(config, 1000.0, {"max_iter": 400})["stacking"]()
        for (_, a), (_, b) in zip(first.estimators, second.estimators):
            self.assertEqual(a.named_steps["model"].get_params(), b.named_steps["model"].get_params())
        with mock.patch.object(models, "tune_ridge", side_effect=AssertionError("later target tuning")), \
                mock.patch.object(models, "tune_hist_gradient_boosting", side_effect=AssertionError("later target tuning")):
            self.stack().fit(self.X, self.y)

    def test_predict_uses_bases_refitted_on_all_fit_rows(self):
        fitted = self.stack().fit(self.X, self.y)
        future = self.X.copy()
        future.index += pd.Timedelta(days=10)
        np.testing.assert_array_equal(fitted.predict(future), np.full(24, 11.5))

    def test_unordered_duplicate_nontime_and_nat_indexes_are_rejected(self):
        bad_indexes = [self.X.index[::-1], pd.Index(range(24)),
                       pd.DatetimeIndex([self.X.index[0]] * 24),
                       pd.DatetimeIndex([pd.NaT, *self.X.index[1:]])]
        for index in bad_indexes:
            X, y = self.X.copy(), self.y.copy()
            X.index = y.index = index
            with self.subTest(index=type(index).__name__):
                with self.assertRaises(ValueError):
                    self.stack().fit(X, y)
                with self.assertRaises(ValueError):
                    chronological_split(X, y)

    def test_misaligned_targets_and_reference_features_are_rejected(self):
        with self.assertRaises(ValueError):
            self.stack().fit(self.X, self.y.iloc[::-1])
        with self.assertRaises(ValueError):
            self.stack().fit(self.X.assign(**{"SO2(GT)": 1.0}), self.y)

    def test_changed_prediction_columns_are_rejected(self):
        fitted = self.stack().fit(self.X, self.y)
        with self.assertRaises(ValueError):
            fitted.predict(self.X.rename(columns={"sensor": "other"}))


if __name__ == "__main__":
    unittest.main()

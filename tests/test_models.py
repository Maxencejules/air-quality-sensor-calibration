import io
import unittest

import support  # noqa: F401  (puts src/ on sys.path)

import numpy as np

from aqcal import models
from aqcal.dataset import load_air_quality
from aqcal.features import CO_SENSOR, CONTINUOUS_COLUMNS, FEATURE_COLUMNS, prepare_dataset
from aqcal.splits import chronological_split


def _prepared(hours=240):
    fixture = support.synthetic_export(hours=hours)
    return prepare_dataset(load_air_quality(io.StringIO(fixture.text)))


class ModelFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = _prepared()
        cls.split = chronological_split(prepared.X, prepared.y)

    def _fit(self, estimator):
        return estimator.fit(self.split.X_train, self.split.y_train)

    def test_every_model_fits_and_predicts_finite_values(self):
        candidates = {
            "mean_baseline": models.mean_baseline(),
            "single_sensor_linear": models.single_sensor_linear(),
            "ridge": models.ridge(1.0),
            "random_forest": models.random_forest(n_estimators=10, seed=0),
            "hist_gradient_boosting": models.hist_gradient_boosting(seed=0, max_iter=20),
            "stacking": models.stacking(n_estimators=10, seed=0, folds=3),
        }
        for name, estimator in candidates.items():
            with self.subTest(model=name):
                predictions = self._fit(estimator).predict(self.split.X_test)
                self.assertEqual(len(predictions), len(self.split.X_test))
                self.assertTrue(np.all(np.isfinite(predictions)))

    def test_single_sensor_model_ignores_every_other_column(self):
        fitted = self._fit(models.single_sensor_linear())
        X = self.split.X_test.copy()
        before = fitted.predict(X)
        rng = np.random.default_rng(0)
        for column in FEATURE_COLUMNS:
            if column != CO_SENSOR:
                X[column] = rng.permutation(X[column].to_numpy())
        np.testing.assert_allclose(fitted.predict(X), before)

    def test_imputer_statistics_come_from_training_rows_only(self):
        fitted = self._fit(models.ridge(1.0))
        imputer = fitted.named_steps["prep"].named_transformers_["continuous"].named_steps["impute"]
        expected = self.split.X_train[list(CONTINUOUS_COLUMNS)].median().to_numpy()
        np.testing.assert_allclose(imputer.statistics_, expected)
        full = self.split.X_train_val[list(CONTINUOUS_COLUMNS)].median().to_numpy()
        self.assertFalse(np.allclose(imputer.statistics_, full))

    def test_missing_features_at_prediction_time_are_imputed(self):
        X = self.split.X_test.copy()
        X.iloc[0, X.columns.get_loc("PT08.S2(NMHC)")] = np.nan
        for estimator in (models.ridge(1.0), models.random_forest(10, 0), models.hist_gradient_boosting(0, max_iter=20)):
            predictions = self._fit(estimator).predict(X)
            self.assertTrue(np.all(np.isfinite(predictions)))

    def test_single_sensor_empty_training_column_is_retained_without_future_data(self):
        X = self.split.X_train.copy()
        X[CO_SENSOR] = np.nan
        fitted = models.single_sensor_linear().fit(X, self.split.y_train)
        imputer = fitted.named_steps["select"].named_transformers_["co_sensor"]
        np.testing.assert_array_equal(imputer.statistics_, [0.0])
        # A sensor first observed in the future cannot create a learned slope.
        later = self.split.X_test.copy()
        later[CO_SENSOR] = 1e9
        np.testing.assert_allclose(fitted.predict(later), self.split.y_train.mean())


class TuningTests(unittest.TestCase):
    def test_searches_return_plain_parameters(self):
        from sklearn.model_selection import TimeSeriesSplit

        prepared = _prepared()
        split = chronological_split(prepared.X, prepared.y)
        folds = TimeSeriesSplit(n_splits=3)

        ridge_info = models.tune_ridge(split.X_train, split.y_train, folds)
        self.assertIn(ridge_info["alpha"], list(models.RIDGE_ALPHAS))
        self.assertGreater(ridge_info["cv_rmse"], 0.0)

        hgb_info = models.tune_hist_gradient_boosting(split.X_train, split.y_train, folds, n_iter=2, seed=0)
        expected = {key.split("__", 1)[1] for key in models.HGB_SEARCH_SPACE}
        self.assertEqual(set(hgb_info["params"]), expected)
        for value in hgb_info["params"].values():
            self.assertIsInstance(value, (int, float))


if __name__ == "__main__":
    unittest.main()

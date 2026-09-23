import tempfile
import unittest
from unittest import mock

import support  # noqa: F401  (puts src/ on sys.path)

from aqcal import experiment
from aqcal.dataset import load_air_quality
from aqcal.experiment import MODEL_ORDER, RunConfig, run_experiment, select_model
from aqcal.features import prepare_dataset
from aqcal.splits import chronological_split


def _entry(val_rmse, test_rmse):
    return {
        "validation": {"rmse": val_rmse, "mae": val_rmse, "r2": 0.0},
        "test": {"rmse": test_rmse, "mae": test_rmse, "r2": 0.0},
    }


# Validation RMSE ranks ridge first; test RMSE ranks stacking first.
DIVERGENT_SCORES = {
    "mean_baseline": _entry(1.50, 1.30),
    "single_sensor_linear": _entry(0.85, 0.73),
    "ridge": _entry(0.60, 0.50),
    "random_forest": _entry(0.70, 0.48),
    "hist_gradient_boosting": _entry(0.75, 0.47),
    "stacking": _entry(0.72, 0.40),
}


class SelectModelTests(unittest.TestCase):
    def test_picks_lowest_validation_rmse_not_lowest_test_rmse(self):
        self.assertEqual(select_model(DIVERGENT_SCORES), "ridge")

    def test_follows_the_validation_scores_when_they_change(self):
        scores = dict(DIVERGENT_SCORES)
        scores["ridge"] = _entry(0.90, 0.50)
        scores["random_forest"] = _entry(0.65, 0.95)
        self.assertEqual(select_model(scores), "random_forest")

    def test_test_scores_never_change_the_choice(self):
        scores = {name: _entry(entry["validation"]["rmse"], 0.0) for name, entry in DIVERGENT_SCORES.items()}
        scores["mean_baseline"] = _entry(1.50, -1.0)
        self.assertEqual(select_model(scores), "ridge")

    def test_tie_goes_to_the_model_listed_first(self):
        scores = {name: _entry(0.5, 0.5) for name in MODEL_ORDER}
        self.assertEqual(select_model(scores), MODEL_ORDER[0])
        self.assertEqual(select_model(scores, order=tuple(reversed(MODEL_ORDER))), MODEL_ORDER[-1])

    def test_names_missing_from_scores_are_skipped(self):
        scores = {"ridge": _entry(0.9, 0.1), "stacking": _entry(0.8, 0.9)}
        self.assertEqual(select_model(scores), "stacking")

    def test_empty_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            select_model({})


class RunExperimentSelectionTests(unittest.TestCase):
    """Patch the scorer so validation and test rankings disagree inside a real run."""

    VALIDATION_BEST = "random_forest"
    TEST_BEST = "mean_baseline"

    def test_run_selects_by_validation_scores(self):
        config = RunConfig(
            cv_splits=3, search_iterations=2, forest_trees=10, importance_repeats=2, n_jobs=1
        )
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = support.write_synthetic(tmp, hours=300)
            prepared = prepare_dataset(load_air_quality(path))
            split = chronological_split(prepared.X, prepared.y, config.train_frac, config.val_frac)

            calls = {"validation": 0, "test": 0}

            def fake_score_all(y_true, y_pred):
                if y_true.index.equals(split.y_val.index):
                    stage, best = "validation", self.VALIDATION_BEST
                elif y_true.index.equals(split.y_test.index):
                    stage, best = "test", self.TEST_BEST
                else:
                    raise AssertionError("scored rows are neither the validation nor the test block")
                name = MODEL_ORDER[calls[stage]]
                calls[stage] += 1
                rmse = 0.1 if name == best else 1.0 + MODEL_ORDER.index(name)
                return {"rmse": rmse, "mae": rmse, "r2": 0.0}

            with mock.patch.object(experiment, "score_all", side_effect=fake_score_all):
                result = run_experiment(path, config)

        self.assertEqual(calls, {"validation": len(MODEL_ORDER), "test": len(MODEL_ORDER)})
        self.assertEqual(result.scores[self.TEST_BEST]["test"]["rmse"], 0.1)
        self.assertEqual(result.selected_model, self.VALIDATION_BEST)


if __name__ == "__main__":
    unittest.main()

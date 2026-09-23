import contextlib
import csv
import io
import json
import math
import os
import tempfile
import unittest

import support  # noqa: F401  (puts src/ on sys.path)

from aqcal.cli import main
from aqcal.experiment import MODEL_ORDER
from aqcal.features import FEATURE_COLUMNS

FAST_ARGS = [
    "--search-iterations", "2",
    "--cv-splits", "3",
    "--forest-trees", "10",
    "--importance-repeats", "2",
    "--n-jobs", "1",
]


def _run(data_path, out_dir):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(["run", "--data", data_path, "--out", out_dir, *FAST_ARGS])
    return code, buffer.getvalue()


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_path, cls.fixture = support.write_synthetic(cls._tmp.name, hours=400)
        cls.out_dir = os.path.join(cls._tmp.name, "outputs")
        cls.code, cls.stdout = _run(cls.data_path, cls.out_dir)
        with open(os.path.join(cls.out_dir, "metrics.json"), encoding="utf-8") as handle:
            cls.metrics = json.load(handle)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_command_succeeds_and_writes_three_files(self):
        self.assertEqual(self.code, 0)
        for name in ("metrics.json", "metrics.csv", "feature_importance.csv"):
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir, name)), name)

    def test_summary_is_printed(self):
        self.assertIn("<- selected", self.stdout)
        for name in MODEL_ORDER:
            self.assertIn(name, self.stdout)

    def test_row_accounting_matches_fixture(self):
        ds = self.metrics["dataset"]
        self.assertEqual(ds["rows_parsed"], self.fixture.hours)
        self.assertEqual(ds["rows_missing_target"], self.fixture.missing_target_rows)
        self.assertEqual(ds["rows_without_sensor_data"], self.fixture.offline_rows)
        self.assertEqual(ds["rows_used"], self.fixture.usable_rows)
        self.assertEqual(ds["features"], list(FEATURE_COLUMNS))
        self.assertEqual(ds["file"], "synthetic.csv")

    def test_periods_are_consecutive(self):
        p = self.metrics["periods"]
        self.assertLess(p["train"]["end"], p["validation"]["start"])
        self.assertLess(p["validation"]["end"], p["test"]["start"])
        total = sum(block["rows"] for block in p.values())
        self.assertEqual(total, self.fixture.usable_rows)

    def test_every_model_has_finite_scores(self):
        scores = self.metrics["scores"]
        self.assertEqual(list(scores), list(MODEL_ORDER))
        for name, entry in scores.items():
            for stage in ("validation", "test"):
                for metric in ("rmse", "mae", "r2"):
                    self.assertTrue(math.isfinite(entry[stage][metric]), f"{name} {stage} {metric}")

    def test_reported_selection_matches_reported_validation_scores(self):
        # Consistency of metrics.json only. On this fixture the validation-best
        # and test-best models can coincide, so the selection rule itself is
        # tested in test_experiment.py with scores where they differ.
        scores = self.metrics["scores"]
        best = min(scores, key=lambda name: scores[name]["validation"]["rmse"])
        self.assertEqual(self.metrics["selected_model"], best)

    def test_sensor_models_beat_the_mean_on_synthetic_signal(self):
        scores = self.metrics["scores"]
        baseline = scores["mean_baseline"]["test"]["rmse"]
        for name in ("single_sensor_linear", "ridge"):
            self.assertLess(scores[name]["test"]["rmse"], baseline, name)

    def test_metrics_csv_has_one_row_per_model(self):
        with open(os.path.join(self.out_dir, "metrics.csv"), newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["model"] for row in rows], list(MODEL_ORDER))
        self.assertIn("test_rmse", rows[0])

    def test_feature_importance_lists_each_feature_once(self):
        with open(os.path.join(self.out_dir, "feature_importance.csv"), newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sorted(row["feature"] for row in rows), sorted(FEATURE_COLUMNS))
        values = [float(row["rmse_increase_mean"]) for row in rows]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_same_seed_gives_identical_metrics(self):
        with tempfile.TemporaryDirectory() as other:
            code, _ = _run(self.data_path, other)
            self.assertEqual(code, 0)
            with open(os.path.join(other, "metrics.json"), encoding="utf-8") as handle:
                again = json.load(handle)
        self.assertEqual(again["scores"], self.metrics["scores"])
        self.assertEqual(again["tuning"], self.metrics["tuning"])

    def test_missing_data_file_is_reported(self):
        with tempfile.TemporaryDirectory() as other, contextlib.redirect_stderr(io.StringIO()) as err:
            code = main(["run", "--data", os.path.join(other, "absent.csv"), "--out", other])
            self.assertFalse(os.path.exists(os.path.join(other, "metrics.json")))
        self.assertEqual(code, 1)
        self.assertIn("error:", err.getvalue())

    def test_invalid_fraction_combination_is_rejected(self):
        with tempfile.TemporaryDirectory() as other, contextlib.redirect_stderr(io.StringIO()):
            code = main(["run", "--data", self.data_path, "--out", other, "--train-frac", "0.9", "--val-frac", "0.2"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

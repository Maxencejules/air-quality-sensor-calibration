import json
import math
from pathlib import Path
import tempfile
import unittest

import support  # noqa: F401
import pandas as pd

from aqcal.experiment import ExperimentResult, MODEL_ORDER
from aqcal.metrics import score_all
from aqcal.report import write_outputs


class ConstantTargetReportTests(unittest.TestCase):
    def test_undefined_r_squared_is_written_as_standard_json_null(self):
        scores = score_all([3.0, 3.0], [3.0, 3.0])
        self.assertTrue(math.isnan(scores["r2"]))
        result = ExperimentResult(
            dataset={}, periods={}, tuning={}, selected_model="mean_baseline",
            scores={name: {"validation": scores, "test": scores} for name in MODEL_ORDER},
            importance=pd.DataFrame({"feature": ["sensor"], "rmse_increase_mean": [0.0],
                                     "rmse_increase_std": [0.0]}),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(result, directory)
            for path in paths:
                self.assertNotIn(b"\r\n", Path(path).read_bytes())
            with open(paths[0], encoding="utf-8") as handle:
                text = handle.read()
            self.assertNotIn("NaN", text)
            report = json.loads(text, parse_constant=lambda value: self.fail(value))
            self.assertIsNone(report["scores"]["mean_baseline"]["test"]["r2"])
            self.assertEqual(report["scores"]["mean_baseline"]["test"]["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()

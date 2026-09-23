import io
import unittest

import support  # noqa: F401  (puts src/ on sys.path)

import numpy as np
import pandas as pd

from aqcal.dataset import load_air_quality
from aqcal.features import (
    FEATURE_COLUMNS,
    REFERENCE_COLUMNS,
    SENSOR_COLUMNS,
    TARGET,
    LeakageError,
    check_no_leakage,
    prepare_dataset,
)


def _frame(rows):
    index = pd.DatetimeIndex([r[0] for r in rows], name="timestamp")
    columns = list(REFERENCE_COLUMNS) + list(SENSOR_COLUMNS) + ["T", "RH", "AH"]
    data = [r[1] for r in rows]
    return pd.DataFrame(data, index=index, columns=columns, dtype=float)


NAN = np.nan
#            CO    NMHC  C6H6  NOx   NO2   S1    S2   S3    S4    S5    T    RH  AH
ROWS = [
    ("2004-05-03 07:00", [1.2, NAN, 5.0, 90, 70, 1000, 800, 1100, 1500, 900, 15, 40, 0.7]),
    ("2004-05-03 08:00", [NAN, NAN, 6.0, 95, 72, 1010, 810, 1090, 1510, 910, 16, 41, 0.7]),
    ("2004-05-03 09:00", [2.0, NAN, 7.0, 99, 75, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN]),
    ("2004-05-04 10:00", [2.4, NAN, 8.0, 99, 75, 1100, NAN, 1000, 1600, 980, 18, 38, 0.8]),
]


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.prepared = prepare_dataset(_frame(ROWS))

    def test_feature_columns_exclude_every_reference_analyser(self):
        self.assertEqual(list(self.prepared.X.columns), list(FEATURE_COLUMNS))
        for column in REFERENCE_COLUMNS:
            self.assertNotIn(column, self.prepared.X.columns)
        self.assertFalse(any(c.endswith("(GT)") for c in self.prepared.X.columns))
        self.assertEqual(self.prepared.y.name, TARGET)

    def test_rows_without_target_or_without_sensor_data_are_dropped(self):
        self.assertEqual(self.prepared.rows_parsed, 4)
        self.assertEqual(self.prepared.rows_missing_target, 1)
        self.assertEqual(self.prepared.rows_without_sensor_data, 1)
        self.assertEqual(self.prepared.rows_used, 2)
        self.assertEqual(list(self.prepared.y), [1.2, 2.4])

    def test_partial_gaps_are_left_for_the_pipeline_imputer(self):
        last = self.prepared.X.iloc[-1]
        self.assertTrue(np.isnan(last["PT08.S2(NMHC)"]))
        self.assertEqual(last["PT08.S1(CO)"], 1100)

    def test_calendar_features(self):
        self.assertEqual(list(self.prepared.X["hour"]), [7, 10])
        # 3 May 2004 was a Monday (0), 4 May a Tuesday (1).
        self.assertEqual(list(self.prepared.X["day_of_week"]), [0, 1])

    def test_missing_input_column_is_reported(self):
        with self.assertRaises(ValueError):
            prepare_dataset(_frame(ROWS).drop(columns=["PT08.S3(NOx)"]))

    def test_real_layout_through_loader(self):
        fixture = support.synthetic_export(hours=120)
        prepared = prepare_dataset(load_air_quality(io.StringIO(fixture.text)))
        self.assertEqual(prepared.rows_used, fixture.usable_rows)
        self.assertFalse(prepared.y.isna().any())


class LeakageGuardTests(unittest.TestCase):
    def test_accepts_the_feature_set(self):
        check_no_leakage(FEATURE_COLUMNS)

    def test_rejects_each_reference_column(self):
        for column in REFERENCE_COLUMNS:
            with self.subTest(column=column):
                with self.assertRaises(LeakageError):
                    check_no_leakage(list(FEATURE_COLUMNS) + [column])

    def test_rejects_any_ground_truth_suffix(self):
        with self.assertRaises(LeakageError):
            check_no_leakage(["PT08.S1(CO)", "SO2(GT)"])

    def test_leakage_error_is_a_value_error(self):
        self.assertTrue(issubclass(LeakageError, ValueError))


if __name__ == "__main__":
    unittest.main()

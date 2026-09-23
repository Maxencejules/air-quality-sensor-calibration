import io
import math
import unittest

import support  # noqa: F401  (puts src/ on sys.path)

import pandas as pd

from aqcal.dataset import load_air_quality

HEADER = (
    "Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC);NOx(GT);"
    "PT08.S3(NOx);NO2(GT);PT08.S4(NO2);PT08.S5(O3);T;RH;AH;;"
)

# Three hours given out of order, a negative temperature, -200 markers in
# several columns, the trailing ';;' on every line and a separator-only line.
FIXTURE = "\r\n".join(
    [
        HEADER,
        "01/02/2005;10.00.00;1,5;1100;-200;7,1;900;120;950;100;1500;1000;-1,9;55,0;0,6612;;",
        "01/02/2005;11.00.00;-200;1150;-200;7,4;910;-200;940;-200;1510;1040;-200;53,5;0,6571;;",
        "01/02/2005;09.00.00;2,25;1210;88;9,0;939;131;1140;114;1555;1074;11,9;54,0;0,7502;;",
        ";;;;;;;;;;;;;;;;",
        "",
    ]
)


class LoadAirQualityTests(unittest.TestCase):
    def setUp(self):
        self.frame = load_air_quality(io.StringIO(FIXTURE))

    def test_trailing_empty_columns_and_blank_lines_are_dropped(self):
        self.assertEqual(len(self.frame), 3)
        self.assertEqual(len(self.frame.columns), 13)
        self.assertFalse(any(str(c).startswith("Unnamed") for c in self.frame.columns))
        self.assertNotIn("Date", self.frame.columns)
        self.assertNotIn("Time", self.frame.columns)

    def test_decimal_commas_are_parsed(self):
        row = self.frame.loc[pd.Timestamp("2005-02-01 10:00")]
        self.assertAlmostEqual(row["CO(GT)"], 1.5)
        self.assertAlmostEqual(row["C6H6(GT)"], 7.1)
        self.assertAlmostEqual(row["T"], -1.9)
        self.assertAlmostEqual(row["AH"], 0.6612)
        early = self.frame.loc[pd.Timestamp("2005-02-01 09:00")]
        self.assertAlmostEqual(early["CO(GT)"], 2.25)

    def test_minus_200_becomes_nan_but_other_negatives_survive(self):
        row = self.frame.loc[pd.Timestamp("2005-02-01 11:00")]
        for column in ("CO(GT)", "NMHC(GT)", "NOx(GT)", "NO2(GT)", "T"):
            self.assertTrue(math.isnan(row[column]), column)
        self.assertAlmostEqual(row["PT08.S1(CO)"], 1150.0)
        self.assertFalse((self.frame == -200).any().any())
        self.assertLess(self.frame["T"].min(), 0)

    def test_index_is_day_first_and_sorted(self):
        self.assertIsInstance(self.frame.index, pd.DatetimeIndex)
        self.assertTrue(self.frame.index.is_monotonic_increasing)
        self.assertEqual(
            list(self.frame.index),
            [pd.Timestamp("2005-02-01 09:00"), pd.Timestamp("2005-02-01 10:00"), pd.Timestamp("2005-02-01 11:00")],
        )

    def test_all_measurements_are_float(self):
        self.assertTrue(all(dtype == float for dtype in self.frame.dtypes))

    def test_duplicate_timestamp_is_rejected(self):
        lines = FIXTURE.split("\r\n")
        duplicated = "\r\n".join([lines[0], lines[1], lines[1]])
        with self.assertRaises(ValueError):
            load_air_quality(io.StringIO(duplicated))

    def test_text_in_a_measurement_is_rejected(self):
        broken = FIXTURE.replace("1100", "abc", 1)
        with self.assertRaises(ValueError):
            load_air_quality(io.StringIO(broken))

    def test_synthetic_export_round_trip(self):
        fixture = support.synthetic_export(hours=48)
        frame = load_air_quality(io.StringIO(fixture.text))
        self.assertEqual(len(frame), 48)
        self.assertEqual(int(frame["CO(GT)"].isna().sum()), fixture.missing_target_rows)


if __name__ == "__main__":
    unittest.main()

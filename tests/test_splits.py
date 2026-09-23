import unittest

import support  # noqa: F401  (puts src/ on sys.path)

import numpy as np
import pandas as pd

from aqcal.splits import chronological_split, split_sizes


def _series(n, start="2004-03-10 18:00"):
    index = pd.date_range(start, periods=n, freq="h", name="timestamp")
    X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) * 2}, index=index)
    y = pd.Series(np.arange(n, dtype=float), index=index, name="target")
    return X, y


class ChronologicalSplitTests(unittest.TestCase):
    def setUp(self):
        X, y = _series(200)
        # Drop a few hours so the index has gaps, as the real data does.
        keep = np.ones(len(X), dtype=bool)
        keep[[10, 11, 57, 150]] = False
        self.X, self.y = X[keep], y[keep]
        self.split = chronological_split(self.X, self.y, train_frac=0.7, val_frac=0.15)

    def test_blocks_cover_every_row_once(self):
        s = self.split
        self.assertEqual(len(s.X_train) + len(s.X_val) + len(s.X_test), len(self.X))
        joined = s.X_train.index.append(s.X_val.index).append(s.X_test.index)
        self.assertTrue(joined.equals(self.X.index))

    def test_blocks_do_not_overlap(self):
        s = self.split
        self.assertEqual(len(s.X_train.index.intersection(s.X_val.index)), 0)
        self.assertEqual(len(s.X_train.index.intersection(s.X_test.index)), 0)
        self.assertEqual(len(s.X_val.index.intersection(s.X_test.index)), 0)

    def test_blocks_are_in_time_order(self):
        s = self.split
        self.assertLess(s.X_train.index.max(), s.X_val.index.min())
        self.assertLess(s.X_val.index.max(), s.X_test.index.min())
        for block in (s.X_train, s.X_val, s.X_test):
            self.assertTrue(block.index.is_monotonic_increasing)

    def test_targets_stay_aligned_with_features(self):
        s = self.split
        for X_part, y_part in ((s.X_train, s.y_train), (s.X_val, s.y_val), (s.X_test, s.y_test)):
            self.assertTrue(X_part.index.equals(y_part.index))
            np.testing.assert_array_equal(X_part["a"].to_numpy(), y_part.to_numpy())

    def test_sizes_follow_fractions(self):
        n = len(self.X)
        self.assertEqual(len(self.split.X_train), round(n * 0.7))
        self.assertEqual(len(self.split.X_val), round(n * 0.15))

    def test_train_val_concatenation_precedes_test(self):
        s = self.split
        self.assertEqual(len(s.X_train_val), len(s.X_train) + len(s.X_val))
        self.assertLess(s.X_train_val.index.max(), s.X_test.index.min())

    def test_periods_report(self):
        periods = self.split.periods()
        self.assertEqual(set(periods), {"train", "validation", "test"})
        self.assertEqual(periods["train"]["start"], self.X.index[0].isoformat())
        self.assertEqual(periods["test"]["end"], self.X.index[-1].isoformat())

    def test_unsorted_input_is_rejected(self):
        X, y = _series(50)
        with self.assertRaises(ValueError):
            chronological_split(X.iloc[::-1], y.iloc[::-1])

    def test_misaligned_input_is_rejected(self):
        X, y = _series(50)
        with self.assertRaises(ValueError):
            chronological_split(X, y.iloc[1:])


class SplitSizeTests(unittest.TestCase):
    def test_counts_add_up(self):
        self.assertEqual(split_sizes(100, 0.7, 0.15), (70, 15, 15))
        self.assertEqual(sum(split_sizes(7344, 0.7, 0.15)), 7344)

    def test_invalid_fractions(self):
        for train, val in ((0.0, 0.1), (1.0, 0.1), (0.8, 0.2), (0.9, 0.15)):
            with self.subTest(train=train, val=val):
                with self.assertRaises(ValueError):
                    split_sizes(100, train, val)

    def test_too_few_rows(self):
        with self.assertRaises(ValueError):
            split_sizes(3, 0.7, 0.15)


if __name__ == "__main__":
    unittest.main()

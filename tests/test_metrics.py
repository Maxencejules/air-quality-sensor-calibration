import math
import unittest

import support  # noqa: F401  (puts src/ on sys.path)

from aqcal.metrics import mae, r_squared, rmse, score_all


class HandComputedMetricTests(unittest.TestCase):
    # y = [1, 2, 3, 4], prediction = [1, 2, 4, 2]
    # errors      = [0, 0, -1, 2]
    # squared     = [0, 0, 1, 4]   -> SSE = 5, MSE = 1.25, RMSE = sqrt(1.25)
    # absolute    = [0, 0, 1, 2]   -> MAE = 0.75
    # mean(y) = 2.5 -> SS_tot = 2.25 + 0.25 + 0.25 + 2.25 = 5 -> R^2 = 1 - 5/5 = 0
    y = [1.0, 2.0, 3.0, 4.0]
    p = [1.0, 2.0, 4.0, 2.0]

    def test_rmse(self):
        self.assertAlmostEqual(rmse(self.y, self.p), math.sqrt(1.25), places=12)

    def test_mae(self):
        self.assertAlmostEqual(mae(self.y, self.p), 0.75, places=12)

    def test_r_squared(self):
        self.assertAlmostEqual(r_squared(self.y, self.p), 0.0, places=12)

    def test_second_case(self):
        # y = [2, 4, 6], prediction = [3, 4, 5]: errors [-1, 0, 1]
        # MSE = 2/3, MAE = 2/3, SS_tot = 4 + 0 + 4 = 8, R^2 = 1 - 2/8 = 0.75
        y, p = [2.0, 4.0, 6.0], [3.0, 4.0, 5.0]
        self.assertAlmostEqual(rmse(y, p), math.sqrt(2.0 / 3.0), places=12)
        self.assertAlmostEqual(mae(y, p), 2.0 / 3.0, places=12)
        self.assertAlmostEqual(r_squared(y, p), 0.75, places=12)

    def test_perfect_prediction(self):
        scores = score_all(self.y, self.y)
        self.assertEqual(scores, {"rmse": 0.0, "mae": 0.0, "r2": 1.0})

    def test_predicting_the_mean_gives_zero_r_squared(self):
        self.assertAlmostEqual(r_squared(self.y, [2.5] * 4), 0.0, places=12)

    def test_worse_than_mean_is_negative(self):
        self.assertLess(r_squared(self.y, [4.0, 3.0, 2.0, 1.0]), 0.0)

    def test_constant_target_gives_nan_r_squared(self):
        self.assertTrue(math.isnan(r_squared([3.0, 3.0], [3.0, 2.0])))

    def test_length_mismatch_and_empty_input(self):
        with self.assertRaises(ValueError):
            rmse([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            mae([], [])


if __name__ == "__main__":
    unittest.main()

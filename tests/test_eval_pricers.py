import unittest

from scripts import eval_pricers


class SummarizeTests(unittest.TestCase):
    def test_reports_accuracy_and_bias(self):
        # Two overshoots, one undershoot against a $100 truth.
        guesses = [120.0, 130.0, 90.0]
        truths = [100.0, 100.0, 100.0]

        stats = eval_pricers.summarize(guesses, truths)

        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mae"], (20 + 30 + 10) / 3)
        self.assertAlmostEqual(stats["bias"], (20 + 30 - 10) / 3)  # net upward bias
        self.assertAlmostEqual(stats["over_rate"], 2 / 3)  # 2 of 3 estimates above truth

    def test_unbiased_estimates_report_zero_bias(self):
        stats = eval_pricers.summarize([90.0, 110.0], [100.0, 100.0])

        self.assertAlmostEqual(stats["bias"], 0.0)
        self.assertAlmostEqual(stats["over_rate"], 0.5)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            eval_pricers.summarize([], [])

    def test_format_summary_line(self):
        line = eval_pricers.format_summary_line(
            {"n": 2, "mae": 10.0, "rmse": 12.5, "bias": -3.25, "over_rate": 0.5}
        )

        self.assertEqual(
            line,
            "MAE: $10.00   RMSE: $12.50   Bias: -$3.25   Over-prediction: 50%   n=2",
        )

    def test_threshold_violations_report_metric_names(self):
        stats = {"mae": 25.0, "bias": 12.0, "over_rate": 0.8}

        violations = eval_pricers.threshold_violations(
            stats,
            max_mae=20.0,
            max_abs_bias=10.0,
            max_over_rate=0.75,
        )

        self.assertEqual(
            violations,
            [
                "mae 25.00 > max_mae 20.00",
                "abs(bias) 12.00 > max_abs_bias 10.00",
                "over_rate 0.80 > max_over_rate 0.75",
            ],
        )

    def test_threshold_violations_ignore_unset_limits(self):
        stats = {"mae": 25.0, "bias": 12.0, "over_rate": 0.8}

        self.assertEqual(eval_pricers.threshold_violations(stats), [])


if __name__ == "__main__":
    unittest.main()

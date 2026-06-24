import types
import unittest

from app.usage import PRICING, UsageTracker


class UsageTrackerTests(unittest.TestCase):
    def test_records_tokens_and_estimates_cost(self):
        tracker = UsageTracker()
        usage = types.SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        tracker.record("gpt-4o-mini", usage)
        in_rate, out_rate = PRICING["gpt-4o-mini"]
        # cost uses the configured LLM_MODEL's rate; default config is gpt-4o-mini.
        self.assertAlmostEqual(tracker.estimated_cost, in_rate + out_rate, places=6)
        self.assertEqual(tracker.calls, 1)

    def test_none_usage_is_ignored(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o-mini", None)
        self.assertEqual(tracker.calls, 0)
        self.assertEqual(tracker.prompt_tokens, 0)

    def test_unpriced_model_flagged_in_report(self):
        tracker = UsageTracker()
        tracker.record("some-unknown-model", types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        self.assertIn("no price for", tracker.report())

    def test_reset_clears_counters(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o-mini", types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        tracker.reset()
        self.assertEqual(tracker.calls, 0)
        self.assertEqual(tracker.prompt_tokens, 0)


if __name__ == "__main__":
    unittest.main()

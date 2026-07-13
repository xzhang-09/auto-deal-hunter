import types
import unittest

from auto_deal_hunter.infra.usage import PRICING, UsageTracker


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

    def test_merge_folds_in_external_usage(self):
        # Simulates the client merging the MCP server subprocess's token totals into the
        # orchestrator's tracker. The orchestration loop's own call is recorded normally;
        # the subprocess's bulk usage arrives via merge() and must be summed in.
        tracker = UsageTracker()
        tracker.record("gpt-4o-mini", types.SimpleNamespace(prompt_tokens=100, completion_tokens=20))
        tracker.merge(prompt_tokens=900, completion_tokens=180, calls=7)
        self.assertEqual(tracker.prompt_tokens, 1000)
        self.assertEqual(tracker.completion_tokens, 200)
        # 1 orchestration call + 7 batched subprocess calls.
        self.assertEqual(tracker.calls, 8)

    def test_merge_propagates_unpriced_models(self):
        tracker = UsageTracker()
        tracker.merge(prompt_tokens=10, completion_tokens=5, calls=1, unpriced_models=["mystery-model"])
        self.assertIn("no price for: mystery-model", tracker.report())

    def test_reset_clears_counters(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o-mini", types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        tracker.reset()
        self.assertEqual(tracker.calls, 0)
        self.assertEqual(tracker.prompt_tokens, 0)


if __name__ == "__main__":
    unittest.main()

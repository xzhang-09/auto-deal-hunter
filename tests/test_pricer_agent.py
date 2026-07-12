import unittest
import sys
import types


sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = lambda *args, **kwargs: None
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from agents.pricer_agent import PLACEHOLDER_PRICE, PricerAgent
from infra.config import VECTOR_SPACE


def _agent_returning(reply: str) -> PricerAgent:
    """A PricerAgent whose LLM call returns ``reply``, with no real network/embedding deps.

    Bypasses __init__ (which needs OpenAI + a SentenceTransformer) and stubs find_similars so
    price() exercises only the parse-and-guard path."""
    agent = PricerAgent.__new__(PricerAgent)
    agent.MODEL = "test-model"
    agent.find_similars = lambda description: ([], [], [])

    message = types.SimpleNamespace(content=reply)
    choice = types.SimpleNamespace(message=message)
    response = types.SimpleNamespace(choices=[choice], usage=None)
    completions = types.SimpleNamespace(create=lambda **kwargs: response)
    agent.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )
    return agent


class PriceEstimateGuardTests(unittest.TestCase):
    def test_echoed_placeholder_is_rejected(self):
        # The model parroting the prompt's example value must not surface as an estimate.
        agent = _agent_returning(f'{{"price": {PLACEHOLDER_PRICE:.2f}}}')
        with self.assertRaises(ValueError):
            agent.price("a niche product with no good comparables")

    def test_non_positive_estimate_is_rejected(self):
        agent = _agent_returning('{"price": 0}')
        with self.assertRaises(ValueError):
            agent.price("anything")

    def test_real_estimate_passes_through(self):
        agent = _agent_returning('{"price": 50.0}')
        self.assertEqual(agent.price("a normally-priced product"), 50.0)

    def test_placeholder_is_not_a_valid_estimate(self):
        # Guard against a future edit setting the placeholder to a plausible price: the prompt
        # example and the price() guard must stay consistent so an echo is always caught.
        self.assertLessEqual(PLACEHOLDER_PRICE, 0)


class RetrievalConfidenceTests(unittest.TestCase):
    def test_no_neighbors_is_zero_confidence(self):
        self.assertEqual(PricerAgent._retrieval_confidence([]), 0.0)

    def test_close_nearest_neighbor_is_high_confidence(self):
        # Nearest cosine distance 0.1 -> confidence 0.9.
        self.assertAlmostEqual(PricerAgent._retrieval_confidence([0.1, 0.4, 0.6]), 0.9)

    def test_distant_nearest_neighbor_is_low_confidence(self):
        # Near-orthogonal nearest neighbor (distance 0.95) -> confidence 0.05.
        self.assertAlmostEqual(PricerAgent._retrieval_confidence([0.95, 1.2]), 0.05)

    def test_confidence_is_clamped_to_unit_interval(self):
        self.assertEqual(PricerAgent._retrieval_confidence([1.7]), 0.0)  # distance > 1
        self.assertEqual(PricerAgent._retrieval_confidence([-0.2]), 1.0)  # distance < 0

    def test_estimate_with_confidence_returns_pair(self):
        agent = _agent_returning('{"price": 80.0}')
        agent.find_similars = lambda description: (["c"], [70.0], [0.2, 0.5])
        estimate, confidence = agent.estimate_with_confidence("a product")
        # Raw 80 shrunk toward the comparable median 70 at confidence 0.8.
        self.assertAlmostEqual(estimate, 0.8 * 80.0 + 0.2 * 70.0)
        self.assertAlmostEqual(confidence, 0.8)


class ShrinkEstimateTests(unittest.TestCase):
    def test_high_confidence_keeps_estimate_close_to_raw(self):
        shrunk = PricerAgent._shrink_estimate(100.0, [60.0, 80.0, 90.0], 0.95)
        self.assertAlmostEqual(shrunk, 0.95 * 100.0 + 0.05 * 80.0)

    def test_low_confidence_pulls_toward_comparable_median(self):
        # A weak RAG match must not win the scan on a wild high guess: the neighbors dominate.
        shrunk = PricerAgent._shrink_estimate(1400.0, [400.0, 450.0, 500.0], 0.2)
        self.assertAlmostEqual(shrunk, 0.2 * 1400.0 + 0.8 * 450.0)

    def test_no_comparables_keeps_raw_estimate(self):
        self.assertEqual(PricerAgent._shrink_estimate(120.0, [], 0.0), 120.0)

    def test_missing_distances_keep_raw_estimate(self):
        # Distances absent (legacy store metadata) -> confidence 0 signals missing data, not a
        # bad match; the pipeline must not replace the LLM estimate with the bare median.
        agent = _agent_returning('{"price": 80.0}')
        agent.find_similars = lambda description: (["c"], [10.0], [])
        estimate, confidence = agent.estimate_with_confidence("a product")
        self.assertEqual(estimate, 80.0)
        self.assertEqual(confidence, 0.0)


class DistanceSpaceGuardTests(unittest.TestCase):
    @staticmethod
    def _check(metadata):
        agent = PricerAgent.__new__(PricerAgent)
        collection = types.SimpleNamespace(metadata=metadata)
        agent._check_distance_space(collection)

    def test_matching_space_passes(self):
        self._check({"hnsw:space": VECTOR_SPACE})  # does not raise

    def test_wrong_space_raises(self):
        with self.assertRaises(ValueError):
            self._check({"hnsw:space": "l2"})

    def test_store_without_space_warns_but_passes(self):
        # No hnsw:space key: warn-and-allow, not raise.
        self._check({"embedding_model": "x"})
        self._check(None)


class PricerAgentParsingTests(unittest.TestCase):
    def test_get_price_accepts_plain_price(self):
        self.assertEqual(PricerAgent.get_price("$1,299.99"), 1299.99)

    def test_get_price_accepts_json_price(self):
        self.assertEqual(PricerAgent.get_price('{"price": 249.5}'), 249.5)

    def test_get_price_rejects_ranges(self):
        with self.assertRaises(ValueError):
            PricerAgent.get_price("$90 to $120")

    def test_get_price_rejects_missing_price(self):
        with self.assertRaises(ValueError):
            PricerAgent.get_price("unknown")


class ComparableNormalizationTests(unittest.TestCase):
    def test_single_unit_comparables_are_unchanged(self):
        docs = ["Widget A", "Widget B"]
        metas = [{"price": 100.0, "quantity": 1}, {"price": 50.0}]
        out_docs, out_prices = PricerAgent._to_comparables(docs, metas)
        self.assertEqual(out_docs, docs)
        self.assertEqual(out_prices, [100.0, 50.0])

    def test_multipack_price_normalized_to_per_unit(self):
        docs = ["AAA Batteries 36-Pack"]
        metas = [{"price": 18.0, "quantity": 36}]
        out_docs, out_prices = PricerAgent._to_comparables(docs, metas)
        self.assertEqual(out_prices, [0.5])
        self.assertIn("per-unit", out_docs[0])
        self.assertIn("pack of 36", out_docs[0])

    def test_missing_quantity_defaults_to_single(self):
        out_docs, out_prices = PricerAgent._to_comparables(["X"], [{"price": 200.0}])
        self.assertEqual(out_prices, [200.0])
        self.assertEqual(out_docs, ["X"])


if __name__ == "__main__":
    unittest.main()

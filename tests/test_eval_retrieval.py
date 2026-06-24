import sys
import types
import unittest

# eval_retrieval imports sentence_transformers at call time, but the metric helpers under
# test don't; stub it so importing the module is cheap and offline.
sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = lambda *a, **k: None
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from scripts.eval_retrieval import aggregate, retrieval_metrics


class RetrievalMetricsTests(unittest.TestCase):
    def test_perfect_category_match(self):
        item = {"category": "Electronics", "price": 100.0}
        neighbors = [
            {"category": "Electronics", "price": 90.0},
            {"category": "Electronics", "price": 110.0},
        ]
        m = retrieval_metrics(neighbors, item)
        self.assertEqual(m["category_precision"], 1.0)
        self.assertEqual(m["hit"], 1.0)
        # median neighbor price = 100 -> 0% error
        self.assertAlmostEqual(m["price_ape"], 0.0)

    def test_partial_category_and_price_error(self):
        item = {"category": "Electronics", "price": 100.0}
        neighbors = [
            {"category": "Electronics", "price": 50.0},
            {"category": "Toys", "price": 50.0},
        ]
        m = retrieval_metrics(neighbors, item)
        self.assertEqual(m["category_precision"], 0.5)
        self.assertEqual(m["hit"], 1.0)
        self.assertAlmostEqual(m["price_ape"], 0.5)  # median 50 vs 100

    def test_no_category_match_is_miss(self):
        item = {"category": "Electronics", "price": 100.0}
        neighbors = [{"category": "Toys", "price": 100.0}]
        m = retrieval_metrics(neighbors, item)
        self.assertEqual(m["hit"], 0.0)
        self.assertEqual(m["category_precision"], 0.0)

    def test_aggregate_averages(self):
        per_query = [
            {"category_precision": 1.0, "hit": 1.0, "price_ape": 0.0},
            {"category_precision": 0.0, "hit": 0.0, "price_ape": 0.4},
        ]
        agg = aggregate(per_query)
        self.assertEqual(agg["n"], 2)
        self.assertAlmostEqual(agg["category_precision"], 0.5)
        self.assertAlmostEqual(agg["hit_rate"], 0.5)
        self.assertAlmostEqual(agg["price_mape"], 0.2)


if __name__ == "__main__":
    unittest.main()

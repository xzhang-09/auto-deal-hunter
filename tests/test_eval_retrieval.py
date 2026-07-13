import unittest

from auto_deal_hunter.evaluation.retrieval import aggregate, retrieval_metrics


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

    def test_multipack_prices_normalized_to_per_unit(self):
        # A 36-pack neighbor at $72 should count as $2/unit, matching a single-unit item.
        item = {"category": "Electronics", "price": 2.0}
        neighbors = [{"category": "Electronics", "price": 72.0, "quantity": 36}]
        m = retrieval_metrics(neighbors, item)
        self.assertAlmostEqual(m["price_ape"], 0.0)

    def test_near_zero_truth_price_excluded_from_ape(self):
        # A $0.06/unit truth price makes APE explode on any neighbor movement (one such row
        # once accounted for an entire rerank A/B regression), so it is not APE-scorable.
        item = {"category": "Electronics", "price": 12.0, "quantity": 200}
        neighbors = [{"category": "Electronics", "price": 5.0}]
        m = retrieval_metrics(neighbors, item)
        self.assertIsNone(m["price_ape"])
        # Category metrics still count the row.
        self.assertEqual(m["category_precision"], 1.0)
        self.assertEqual(m["hit"], 1.0)

    def test_multipack_query_item_normalized(self):
        # A 10-pack query item at $20 ($2/unit) vs single-unit $2 neighbors -> 0% error.
        item = {"category": "Electronics", "price": 20.0, "quantity": 10}
        neighbors = [{"category": "Electronics", "price": 2.0}]
        m = retrieval_metrics(neighbors, item)
        self.assertAlmostEqual(m["price_ape"], 0.0)

    def test_aggregate_averages(self):
        per_query = [
            {"category_precision": 1.0, "hit": 1.0, "price_ape": 0.0},
            {"category_precision": 0.0, "hit": 0.0, "price_ape": 0.4},
        ]
        agg = aggregate(per_query)
        self.assertEqual(agg["n"], 2)
        self.assertEqual(agg["n_ape"], 2)
        self.assertAlmostEqual(agg["category_precision"], 0.5)
        self.assertAlmostEqual(agg["hit_rate"], 0.5)
        self.assertAlmostEqual(agg["price_mape"], 0.2)

    def test_aggregate_counts_ape_scorable_rows(self):
        # Rows whose APE was excluded (near-zero truth price) count in n but not n_ape.
        per_query = [
            {"category_precision": 1.0, "hit": 1.0, "price_ape": 0.3},
            {"category_precision": 1.0, "hit": 1.0, "price_ape": None},
        ]
        agg = aggregate(per_query)
        self.assertEqual(agg["n"], 2)
        self.assertEqual(agg["n_ape"], 1)
        self.assertAlmostEqual(agg["price_mape"], 0.3)

    def test_median_ape_is_robust_to_outlier(self):
        # One junk-cheap item with a huge APE must not define the headline metric: the median
        # stays put while the mean is dragged up. Mirrors the $0.01-row blowup seen in practice.
        per_query = [
            {"category_precision": 1.0, "hit": 1.0, "price_ape": 0.3},
            {"category_precision": 1.0, "hit": 1.0, "price_ape": 0.3},
            {"category_precision": 1.0, "hit": 1.0, "price_ape": 0.3},
            {"category_precision": 1.0, "hit": 1.0, "price_ape": 1000.0},
        ]
        agg = aggregate(per_query)
        self.assertAlmostEqual(agg["price_median_ape"], 0.3)
        self.assertGreater(agg["price_mape"], 100.0)


if __name__ == "__main__":
    unittest.main()

import types
import unittest

import app.pipeline as pipeline_module
from app.pipeline import DealPipeline
from domain.deal import Deal, DealSelection


class FakePricer:
    """Returns a preset (estimate, confidence) per description, or raises a preset exception."""

    def __init__(self, by_description):
        self.by_description = by_description

    def estimate_with_confidence(self, description):
        value = self.by_description[description]
        if isinstance(value, Exception):
            raise value
        return value


def _deal(description, price, url, list_price=None):
    return Deal(product_description=description, price=price, list_price=list_price, url=url)


class DealPipelineTests(unittest.TestCase):
    def setUp(self):
        # Pin the confidence gate so the tests don't depend on an env override.
        self._original_threshold = pipeline_module.RAG_MIN_CONFIDENCE
        pipeline_module.RAG_MIN_CONFIDENCE = 0.15
        self.addCleanup(setattr, pipeline_module, "RAG_MIN_CONFIDENCE", self._original_threshold)

    def _pipeline(self, deals, estimates, sent):
        pipeline = DealPipeline(collection=object())
        selection = DealSelection(deals=deals) if deals else None
        pipeline._scanner = types.SimpleNamespace(scan=lambda memory: selection)
        pipeline._pricer = FakePricer(estimates)
        pipeline._messenger = types.SimpleNamespace(notify=lambda *args: sent.append(args))
        return pipeline

    def test_no_deals_returns_none_and_no_push(self):
        sent = []
        pipeline = self._pipeline([], {}, sent)
        memory, best = pipeline.run([])
        self.assertIsNone(best)
        self.assertEqual(sent, [])

    def test_selects_highest_capped_savings_and_notifies(self):
        sent = []
        deals = [
            _deal("A", 50.0, "https://x.test/1.html", list_price=100.0),  # discount 50
            _deal("B", 80.0, "https://x.test/2.html", list_price=100.0),  # discount 20
        ]
        estimates = {"A": (100.0, 0.9), "B": (100.0, 0.9)}
        pipeline = self._pipeline(deals, estimates, sent)

        memory, best = pipeline.run([])

        self.assertEqual(best.deal.url, "https://x.test/1.html")
        self.assertEqual(best.retrieval_confidence, 0.9)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][3], "https://x.test/1.html")  # notify(..., url)

    def test_low_confidence_best_is_saved_but_not_pushed(self):
        sent = []
        deals = [_deal("A", 50.0, "https://x.test/1.html", list_price=100.0)]
        estimates = {"A": (100.0, 0.05)}  # below the 0.15 gate
        pipeline = self._pipeline(deals, estimates, sent)

        memory, best = pipeline.run([])

        self.assertIsNotNone(best)  # still returned -> caller saves it
        self.assertEqual(sent, [])  # but no push

    def test_unusable_estimate_is_skipped(self):
        sent = []
        deals = [
            _deal("A", 50.0, "https://x.test/1.html", list_price=100.0),
            _deal("B", 80.0, "https://x.test/2.html", list_price=100.0),
        ]
        estimates = {"A": ValueError("no usable estimate"), "B": (100.0, 0.9)}
        pipeline = self._pipeline(deals, estimates, sent)

        memory, best = pipeline.run([])

        self.assertEqual(best.deal.url, "https://x.test/2.html")  # A skipped, B wins
        self.assertEqual(len(sent), 1)

    def test_no_positive_discount_returns_none(self):
        sent = []
        # Estimate at/below price -> no genuine bargain -> best_opportunity returns None.
        deals = [_deal("A", 100.0, "https://x.test/1.html", list_price=100.0)]
        estimates = {"A": (100.0, 0.9)}
        pipeline = self._pipeline(deals, estimates, sent)

        memory, best = pipeline.run([])

        self.assertIsNone(best)
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()

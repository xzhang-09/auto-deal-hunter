import unittest

from auto_deal_hunter.domain.deal import Deal, Opportunity
from auto_deal_hunter.scripts.feedback_report import confidence_bucket, summarize_feedback


class FeedbackReportTests(unittest.TestCase):
    def test_confidence_bucket(self):
        self.assertEqual(confidence_bucket(None), "unknown")
        self.assertEqual(confidence_bucket(0.1), "<0.15 (below notify gate)")
        self.assertEqual(confidence_bucket(0.3), "0.15-0.39")
        self.assertEqual(confidence_bucket(0.6), "0.40-0.69")
        self.assertEqual(confidence_bucket(0.9), ">=0.70")

    def test_summarize_feedback_buckets_labels(self):
        rows = [
            (self._opp(0.9, list_price=120.0, estimate=100.0), "good_deal"),
            (self._opp(0.1, list_price=80.0, estimate=100.0), "bad_deal"),
            (self._opp(None, list_price=None, estimate=100.0), None),
        ]

        summary = summarize_feedback(rows)

        self.assertEqual(summary["overall"]["labeled"], 2)
        self.assertEqual(summary["overall"]["precision"], 0.5)
        self.assertEqual(summary["buckets"]["confidence"][">=0.70"]["good"], 1)
        self.assertEqual(summary["buckets"]["confidence"]["<0.15 (below notify gate)"]["bad"], 1)
        self.assertEqual(summary["buckets"]["confidence"]["unknown"]["total"], 1)
        self.assertEqual(summary["buckets"]["overestimate"]["yes"]["bad"], 1)

    def _opp(self, confidence, list_price, estimate):
        return Opportunity(
            deal=Deal(
                product_description="Example product",
                price=50.0,
                list_price=list_price,
                url="https://example.test/deal",
            ),
            estimate=estimate,
            retrieval_confidence=confidence,
        )


if __name__ == "__main__":
    unittest.main()

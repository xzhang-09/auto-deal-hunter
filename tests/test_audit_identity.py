import sys
import types
import unittest


# scripts.audit_identity imports ingest.scraper, which imports feedparser at module load.
feedparser = types.ModuleType("feedparser")
feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
sys.modules.setdefault("feedparser", feedparser)

from auto_deal_hunter.scripts.audit_identity import summarize


def _rec(kind, new_retail=True, title="t"):
    return {
        "title": title,
        "kind": kind,
        "quantity": 1,
        "variant": None,
        "new_retail": new_retail,
        "priceable": True,
        "url": "u",
    }


class SummarizeTests(unittest.TestCase):
    def test_distribution_over_new_retail_only(self):
        records = [
            _rec("none"),
            _rec("single"),
            _rec("multipack"),
            _rec("bundle", new_retail=False),  # excluded from the denominator
        ]
        stats = summarize(records)
        self.assertEqual(stats["scraped"], 4)
        self.assertEqual(stats["new_retail"], 3)
        self.assertAlmostEqual(stats["none_share"], 1 / 3)
        self.assertEqual(stats["kind_counts"]["none"], 1)

    def test_none_samples_are_listed(self):
        stats = summarize([_rec("none", title="Mystery Gadget")])
        self.assertIn("Mystery Gadget", stats["none_samples"])

    def test_classified_samples_exclude_single(self):
        stats = summarize([_rec("multipack", title="AAA 36-Pack"), _rec("single")])
        self.assertIn("AAA 36-Pack", stats["classified_samples"]["multipack"])
        self.assertNotIn("single", stats["classified_samples"])

    def test_empty(self):
        stats = summarize([])
        self.assertEqual(stats["new_retail"], 0)
        self.assertEqual(stats["none_share"], 0.0)


if __name__ == "__main__":
    unittest.main()

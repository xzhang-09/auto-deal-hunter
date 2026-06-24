import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


feedparser = types.ModuleType("feedparser")
feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
sys.modules.setdefault("feedparser", feedparser)

from app.opportunity_store import OpportunityStore
from models.deals import Deal, Opportunity


class OpportunityStoreTests(unittest.TestCase):
    def test_append_and_list_opportunities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")
            opportunity = self._opportunity(url="https://example.test/deal/1.html")

            store.append(opportunity)
            stored = store.list_opportunities()

        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].deal.url, opportunity.deal.url)
        self.assertEqual(stored[0].deal.list_price, 79.99)

    def test_append_is_idempotent_by_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")
            opportunity = self._opportunity(url="https://example.test/deal/1.html")

            store.append(opportunity)
            store.append(opportunity)

            self.assertEqual(len(store.list_opportunities()), 1)

    def test_migrate_from_legacy_memory_json_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            legacy_path = tmp_path / "memory.json"
            legacy_path.write_text(
                json.dumps([self._opportunity(url="https://example.test/deal/1.html").model_dump()])
            )
            store = OpportunityStore(tmp_path / "deals.sqlite")

            store.migrate_from_json(legacy_path)
            store.migrate_from_json(legacy_path)

            self.assertEqual(len(store.list_opportunities()), 1)

    def _opportunity(self, url):
        return Opportunity(
            deal=Deal(
                product_description="Example product",
                price=50.0,
                list_price=79.99,
                url=url,
            ),
            estimate=100.0,
        )


if __name__ == "__main__":
    unittest.main()

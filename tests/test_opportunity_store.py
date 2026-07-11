import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path


feedparser = types.ModuleType("feedparser")
feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
sys.modules.setdefault("feedparser", feedparser)

from core.opportunity_store import OpportunityStore
from domain.deal import Deal, Opportunity


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

    def test_append_upserts_refreshed_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")
            url = "https://example.test/deal/1.html"

            store.append(self._opportunity(url=url, list_price=140.0))
            refreshed = Opportunity(
                deal=Deal(product_description="Example product", price=50.0, list_price=130.0, url=url),
                estimate=100.0,
            )
            store.append(refreshed)

            stored = store.list_opportunities()
            self.assertEqual(len(stored), 1)
            # A re-scrape of the same URL overwrites the stale list price (140 -> 130).
            self.assertEqual(stored[0].deal.list_price, 130.0)

    def test_mark_feedback_records_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")
            opportunity = self._opportunity(url="https://example.test/deal/1.html")
            store.append(opportunity)

            store.mark_feedback(opportunity.deal.url, "good_deal")

            self.assertEqual(store.feedback_counts(), {"good_deal": 1, "bad_deal": 0, "unlabeled": 0})

    def test_list_feedback_rows_pairs_opportunity_and_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")
            opportunity = self._opportunity(url="https://example.test/deal/1.html")
            opportunity.retrieval_confidence = 0.8
            store.append(opportunity)
            store.mark_feedback(opportunity.deal.url, "bad_deal")

            rows = store.list_feedback_rows()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0].deal.url, opportunity.deal.url)
            self.assertEqual(rows[0][0].retrieval_confidence, 0.8)
            self.assertEqual(rows[0][1], "bad_deal")

    def test_mark_feedback_rejects_unknown_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")

            with self.assertRaises(ValueError):
                store.mark_feedback("https://example.test/deal/1.html", "maybe")

    def test_feedback_column_is_added_to_existing_dedup_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "deals.sqlite"
            opportunity = self._opportunity(url="https://example.test/deal/1.html")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE opportunities (dedup_id TEXT PRIMARY KEY, url TEXT NOT NULL, "
                "payload_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "INSERT INTO opportunities (dedup_id, url, payload_json) VALUES (?,?,?)",
                ("1", opportunity.deal.url, json.dumps(opportunity.model_dump())),
            )
            conn.commit()
            conn.close()

            store = OpportunityStore(db_path)
            store.mark_feedback(opportunity.deal.url, "bad_deal")

            self.assertEqual(store.feedback_counts(), {"good_deal": 0, "bad_deal": 1, "unlabeled": 0})

    def test_prune_stale_removes_expired_and_keeps_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")
            store.append(self._opportunity(url="https://example.test/old.html"))
            store.append(self._opportunity(url="https://example.test/new.html"))
            # Backdate one row's updated_at well past any positive TTL.
            with store._connect() as conn:
                conn.execute(
                    "UPDATE opportunities SET updated_at = datetime('now', '-100 hours') "
                    "WHERE url = ?",
                    ("https://example.test/old.html",),
                )

            removed = store.prune_stale(72.0)

            self.assertEqual(removed, 1)
            urls = {opp.deal.url for opp in store.list_opportunities()}
            self.assertEqual(urls, {"https://example.test/new.html"})

    def test_prune_stale_disabled_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")
            store.append(self._opportunity(url="https://example.test/deal/1.html"))
            with store._connect() as conn:
                conn.execute("UPDATE opportunities SET updated_at = datetime('now', '-1000 hours')")

            self.assertEqual(store.prune_stale(0), 0)
            self.assertEqual(len(store.list_opportunities()), 1)

    def test_append_dedups_by_product_id_across_slugs(self):
        # Same DealNews product id (1) under two different slugs/queries must collapse to one
        # row, with the canonical url/payload refreshed to the latest scrape.
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpportunityStore(Path(tmpdir) / "deals.sqlite")
            store.append(self._opportunity(url="https://x.test/Old-Slug/1.html?iref=rss", list_price=140.0))
            store.append(self._opportunity(url="https://x.test/New-Slug/1.html", list_price=130.0))

            stored = store.list_opportunities()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].deal.url, "https://x.test/New-Slug/1.html")
            self.assertEqual(stored[0].deal.list_price, 130.0)

    def test_migrates_url_pk_schema_and_collapses_duplicates(self):
        # A pre-existing store keyed on the full URL held the same product (id 1) twice under
        # different slugs. Opening it must migrate to the deal_id key and keep the freshest row.
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "deals.sqlite"
            older = self._opportunity(url="https://x.test/Old-Slug/1.html", list_price=140.0)
            newer = self._opportunity(url="https://x.test/New-Slug/1.html", list_price=130.0)
            other = self._opportunity(url="https://x.test/Gadget/2.html", list_price=99.0)
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE opportunities (url TEXT PRIMARY KEY, payload_json TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.executemany(
                "INSERT INTO opportunities (url, payload_json, created_at, updated_at) VALUES (?,?,?,?)",
                [
                    (older.deal.url, json.dumps(older.model_dump()), "2026-01-01 00:00:00", "2026-01-01 00:00:00"),
                    (newer.deal.url, json.dumps(newer.model_dump()), "2026-01-02 00:00:00", "2026-01-02 00:00:00"),
                    (other.deal.url, json.dumps(other.model_dump()), "2026-01-01 00:00:00", "2026-01-01 00:00:00"),
                ],
            )
            conn.commit()
            conn.close()

            store = OpportunityStore(db_path)  # triggers migration

            stored = {opp.deal.url: opp for opp in store.list_opportunities()}
            self.assertEqual(len(stored), 2)  # the two id-1 rows collapsed into one
            self.assertIn("https://x.test/New-Slug/1.html", stored)
            self.assertEqual(stored["https://x.test/New-Slug/1.html"].deal.list_price, 130.0)
            self.assertIn("https://x.test/Gadget/2.html", stored)
            # The migrated table is keyed on dedup_id now.
            with store._connect() as conn:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(opportunities)")}
            self.assertIn("dedup_id", cols)

    def test_migrate_does_not_clobber_existing_fresh_row(self):
        # A live scrape refreshes list_price 140 -> 130; a later startup import with
        # list_price=None must NOT overwrite it.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            url = "https://example.test/deal/1.html"
            imported = self._opportunity(url=url)
            imported.deal.list_price = None
            import_path = tmp_path / "memory.json"
            import_path.write_text(json.dumps([imported.model_dump()]))
            store = OpportunityStore(tmp_path / "deals.sqlite")

            store.append(self._opportunity(url=url, list_price=130.0))  # fresh scrape
            store.migrate_from_json(import_path)  # startup import

            stored = store.list_opportunities()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].deal.list_price, 130.0)  # fresh value preserved

    def test_migrate_from_memory_json_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            import_path = tmp_path / "memory.json"
            import_path.write_text(
                json.dumps([self._opportunity(url="https://example.test/deal/1.html").model_dump()])
            )
            store = OpportunityStore(tmp_path / "deals.sqlite")

            store.migrate_from_json(import_path)
            store.migrate_from_json(import_path)

            self.assertEqual(len(store.list_opportunities()), 1)

    def _opportunity(self, url, list_price=79.99):
        return Opportunity(
            deal=Deal(
                product_description="Example product",
                price=50.0,
                list_price=list_price,
                url=url,
            ),
            estimate=100.0,
        )


if __name__ == "__main__":
    unittest.main()

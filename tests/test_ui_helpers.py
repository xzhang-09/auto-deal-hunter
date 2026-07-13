import tempfile
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


agent_mcp = types.ModuleType("auto_deal_hunter.app.mcp_client")
agent_mcp.run_sync = lambda memory: (memory, None)
sys.modules.setdefault("auto_deal_hunter.app.mcp_client", agent_mcp)

chromadb = types.ModuleType("chromadb")
chromadb.PersistentClient = lambda path: None
sys.modules.setdefault("chromadb", chromadb)

feedparser = types.ModuleType("feedparser")
feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
sys.modules.setdefault("feedparser", feedparser)

from auto_deal_hunter.app import ui


class UiHelperTests(unittest.TestCase):
    def setUp(self):
        # Pin the mismatch threshold so the tests don't depend on an env override.
        self._original_ratio = ui.ESTIMATE_MISMATCH_RATIO
        ui.ESTIMATE_MISMATCH_RATIO = 2.0
        self.addCleanup(setattr, ui, "ESTIMATE_MISMATCH_RATIO", self._original_ratio)

    def test_below_estimate_percent(self):
        # 80 paid against a 100 value -> 20% below value; never exceeds 100%.
        self.assertEqual(ui.below_estimate_percent(100.0, 80.0), "20.0%")
        self.assertEqual(ui.below_estimate_percent(100.0, 100.0), "n/a")
        self.assertEqual(ui.below_estimate_percent(0.0, 0.0), "n/a")

    def test_dashboard_shows_saved_deals_and_no_warning_when_ready(self):
        opportunities = [
            self._opportunity(price=50.0, estimate=100.0),
            self._opportunity(price=80.0, estimate=120.0, list_price=100.0),
        ]

        stats = ui.dashboard_stats(opportunities, vector_store_ready=True)
        self.assertEqual(stats["opportunities"], "2")
        self.assertTrue(stats["ready"])

        html = ui.stats_html(stats)
        self.assertIn("2 saved deals", html)
        # The aggregate overestimate counter was removed: per-row ⚠️ n/a marks mismatches,
        # and calibration lives in eval_pricers / feedback_report.
        self.assertNotIn("Est &gt; list price", html)
        self.assertNotIn("setup-warning", html)

    def test_dashboard_shows_setup_warning_when_not_ready(self):
        html = ui.stats_html(ui.dashboard_stats([], vector_store_ready=False))

        self.assertIn("setup-warning", html)
        self.assertIn("build_vector_store", html)

    def test_table_rows_include_list_price(self):
        opportunity = self._opportunity(
            price=50.0,
            estimate=100.0,
            discount=50.0,
            list_price=149.99,
        )

        row = ui.table_for([opportunity])[0]

        # Deal Price, List Price (MSRP), Est. Value, Savings ($), Below Est. %, URL
        self.assertEqual(row[1], "$50.00")
        self.assertEqual(row[2], "$149.99")
        self.assertEqual(row[3], "$100.00")
        self.assertEqual(row[4], "$50.00")
        self.assertEqual(row[5], "50.0%")
        self.assertEqual(row[6], "[View](https://example.test/deal)")

    def test_multipack_rows_show_pack_level_prices(self):
        # Internally a 4-pack is stored per-unit ($4.75/$6.00); the table must show the pack
        # prices the user actually pays ($19/$24), not the per-unit values.
        from auto_deal_hunter.core.identity_policy import per_unit_note

        opp = ui.Opportunity(
            deal=ui.Deal(
                product_description="Bluetooth Tracker Tags" + per_unit_note(4),
                price=4.75,
                list_price=6.0,
                url="https://example.test/deal",
                quantity=4,
            ),
            # Below the 2x-of-list mismatch threshold: this test exercises pack-price
            # display, not the mismatch blanking (covered separately below).
            estimate=10.0,
        )

        row = ui.table_for([opp])[0]

        self.assertEqual(row[1], "$19.00")          # pack deal price
        self.assertEqual(row[2], "$24.00")          # pack list price
        self.assertEqual(row[3], "$40.00")          # pack estimate (10.00 x 4)
        self.assertEqual(row[4], "$5.00")           # total capped savings (1.25 x 4)
        self.assertEqual(row[5], "20.8%")           # below-estimate %, scale-invariant
        self.assertIn("(4-pack)", row[0])
        self.assertNotIn("per-unit", row[0])

    def test_table_rows_do_not_include_confidence_column(self):
        opportunity = self._opportunity(price=50.0, estimate=160.0, list_price=149.99)

        row = ui.table_for([opportunity])[0]

        self.assertEqual(len(row), 10)
        self.assertEqual(row[6], "[View](https://example.test/deal)")
        # Unlabeled rows show plain action emojis.
        self.assertEqual(row[ui.GOOD_COL], "👍")
        self.assertEqual(row[ui.BAD_COL], "👎")
        self.assertEqual(row[ui.ALERT_COL], "🔔")

    def test_table_renders_saved_feedback_labels(self):
        from auto_deal_hunter.core.source_ids import deal_id

        opportunity = self._opportunity(price=50.0, estimate=100.0)
        labels = {deal_id(opportunity.deal.url): "good_deal"}

        row = ui.table_for([opportunity], labels)[0]

        self.assertEqual(row[ui.GOOD_COL], "✅")
        self.assertEqual(row[ui.BAD_COL], "👎")

    def test_handle_cell_action_routes_action_columns(self):
        calls = []
        framework = types.SimpleNamespace(
            memory=[self._opportunity(price=50.0, estimate=100.0)],
            opportunity_store=types.SimpleNamespace(
                mark_feedback=lambda url, label: calls.append((url, label)),
                feedback_map=lambda: {},
            ),
        )

        ui.handle_cell_action(framework, 0, ui.GOOD_COL)
        ui.handle_cell_action(framework, 0, ui.BAD_COL)

        self.assertEqual(
            calls,
            [
                ("https://example.test/deal", "good_deal"),
                ("https://example.test/deal", "bad_deal"),
            ],
        )

    def test_handle_cell_action_ignores_plain_cells(self):
        calls = []
        framework = types.SimpleNamespace(
            memory=[self._opportunity(price=50.0, estimate=100.0)],
            opportunity_store=types.SimpleNamespace(
                mark_feedback=lambda url, label: calls.append((url, label)),
                feedback_map=lambda: {},
            ),
        )

        rows = ui.handle_cell_action(framework, 0, 1)

        self.assertEqual(calls, [])
        self.assertEqual(len(rows), 1)

    def test_mark_feedback_for_row_updates_selected_opportunity(self):
        calls = []
        framework = types.SimpleNamespace(
            memory=[self._opportunity(price=50.0, estimate=100.0)],
            opportunity_store=types.SimpleNamespace(
                mark_feedback=lambda url, label: calls.append((url, label)),
                feedback_map=lambda: {},
            ),
        )

        rows = ui.mark_feedback_for_row(framework, 0, "good_deal")

        self.assertEqual(calls, [("https://example.test/deal", "good_deal")])
        self.assertEqual(rows[0][0], "Example product")

    def test_mark_feedback_for_row_ignores_missing_selection(self):
        calls = []
        framework = types.SimpleNamespace(
            memory=[self._opportunity(price=50.0, estimate=100.0)],
            opportunity_store=types.SimpleNamespace(
                mark_feedback=lambda url, label: calls.append((url, label)),
                feedback_map=lambda: {},
            ),
        )

        rows = ui.mark_feedback_for_row(framework, None, "good_deal")

        self.assertEqual(calls, [])
        self.assertEqual(len(rows), 1)

    def test_alert_for_row_ignores_missing_selection(self):
        framework = types.SimpleNamespace(memory=[])

        self.assertIsNone(ui.alert_for_row(framework, None))

    def test_discount_is_capped_at_list_price(self):
        # estimate above list: value capped at list, so discount = 149.99 - 50, not 160 - 50.
        over = self._opportunity(price=50.0, estimate=160.0, list_price=149.99)
        self.assertAlmostEqual(over.effective_value, 149.99)
        self.assertAlmostEqual(over.discount, 99.99)

        # estimate below list: estimate used as-is.
        under = self._opportunity(price=50.0, estimate=100.0, list_price=149.99)
        self.assertAlmostEqual(under.discount, 50.0)

        # no list price: nothing to cap against, raw estimate used.
        unknown = self._opportunity(price=50.0, estimate=160.0)
        self.assertAlmostEqual(unknown.discount, 110.0)

    def test_legacy_per_unit_note_is_stripped_from_stored_descriptions(self):
        # Records saved before the note wording changed carry the old suffix; the table
        # must strip that too, not only the current per_unit_note output.
        opp = ui.Opportunity(
            deal=ui.Deal(
                product_description="AAA Batteries (per-unit price; sold in packs of 48)",
                price=0.5,
                list_price=0.8125,
                url="https://example.test/deal",
                quantity=48,
            ),
            estimate=0.6,
        )

        row = ui.table_for([opp])[0]

        self.assertEqual(row[0], "AAA Batteries (48-pack)")

    def test_mismatch_row_blanks_estimate_but_keeps_capped_savings(self):
        # Estimate at a multiple of list price (retrieval mismatch): the estimate and the
        # estimate-derived percentage are blanked, but the capped savings stay visible.
        opportunity = self._opportunity(price=24.0, estimate=1003.43, list_price=39.0)

        row = ui.table_for([opportunity])[0]

        self.assertEqual(row[3], "⚠️ n/a")
        self.assertEqual(row[4], "$15.00")  # min(estimate, list) - price
        self.assertEqual(row[5], "n/a")

    def test_is_comparables_mismatch_predicate(self):
        mismatch = self._opportunity(price=24.0, estimate=1003.43, list_price=39.0)
        ordinary = self._opportunity(price=870.0, estimate=1099.0, list_price=1050.0)
        no_list = self._opportunity(price=24.0, estimate=1003.43, list_price=None)

        self.assertTrue(mismatch.is_comparables_mismatch(2.0))
        self.assertFalse(ordinary.is_comparables_mismatch(2.0))  # 1.05x list: normal noise
        self.assertFalse(no_list.is_comparables_mismatch(2.0))  # nothing to compare against

    def test_is_overestimate_flag(self):
        over = self._opportunity(price=50.0, estimate=160.0, list_price=149.99)
        under = self._opportunity(price=50.0, estimate=100.0, list_price=149.99)
        unknown = self._opportunity(price=50.0, estimate=160.0, list_price=None)

        self.assertTrue(over.is_overestimate)
        self.assertFalse(under.is_overestimate)
        self.assertFalse(unknown.is_overestimate)

    def test_vector_store_status_detects_chroma_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            self.assertFalse(ui.vector_store_ready(path))

            (path / "chroma.sqlite3").write_text("")

            self.assertTrue(ui.vector_store_ready(path))

    def test_reference_map_caption_is_available_for_layout(self):
        self.assertIn("3D t-SNE projection", ui.REFERENCE_MAP_CAPTION)
        self.assertIn(".section-caption", ui.APP_CSS)
        self.assertNotIn("margin: -", ui.APP_CSS)

    def test_reference_map_caption_is_declared_after_plot(self):
        source = (ROOT / "auto_deal_hunter" / "app" / "ui.py").read_text()

        self.assertLess(
            source.index("plot = gr.Plot"),
            source.index("gr.Markdown(REFERENCE_MAP_CAPTION)"),
        )

    def _opportunity(self, price, estimate, discount=None, list_price=None):
        # discount is now a computed, list-price-capped property; the kwarg is accepted for
        # call-site compatibility but no longer fed into the model.
        return ui.Opportunity(
            deal=ui.Deal(
                product_description="Example product",
                price=price,
                list_price=list_price,
                url="https://example.test/deal",
            ),
            estimate=estimate,
        )


if __name__ == "__main__":
    unittest.main()

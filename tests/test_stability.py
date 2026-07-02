import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class StabilityTests(unittest.TestCase):
    def test_reformat_escapes_html_but_keeps_color_spans(self):
        from infra.log_utils import BG_BLACK, GREEN, RESET, reformat

        message = f"{BG_BLACK}{GREEN}<script>alert(1)</script>{RESET}"

        result = reformat(message)

        self.assertIn('<span style="color: #00dd00">', result)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", result)
        self.assertNotIn("<script>", result)

    def test_read_memory_allows_sqlite_filename_without_directory(self):
        self._install_deal_framework_stubs()
        module = importlib.import_module("app.orchestrator")

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            self.addCleanup(os.chdir, original_cwd)
            db_file = Path("deals.sqlite")
            framework = module.Orchestrator.__new__(module.Orchestrator)
            framework.DEALS_DB_PATH = str(db_file)

            # read_memory falls back to constructing an OpportunityStore when the instance was
            # built without __init__; a bare filename (no parent dir) must not raise.
            self.assertEqual(framework.read_memory(), [])
            self.assertTrue(db_file.exists())

    def test_scraped_deal_fetch_skips_detail_request_failures(self):
        self._install_model_stubs()
        from ingest.scraper import ScrapedDeal

        entry = {
            "title": "Working deal",
            "summary": "<div class='snippet summary'>summary</div>",
            "links": [{"href": "https://example.test/deal/1.html"}],
        }
        broken = {
            "title": "Broken deal",
            "summary": "<div class='snippet summary'>summary</div>",
            "links": [{"href": "https://example.test/deal/2.html"}],
        }

        def fake_get(url, timeout=None):
            if url.endswith("2.html"):
                raise RuntimeError("network failed")
            return types.SimpleNamespace(
                content=b"<div class='content-section'>Details Features Specs</div>",
                raise_for_status=lambda: None,
            )

        with (
            patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "0"}),
            patch("ingest.scraper.FEEDS", ["https://example.test/rss"]),
            patch("ingest.scraper.feedparser.parse") as parse,
            patch("ingest.scraper.requests.get", side_effect=fake_get),
        ):
            parse.return_value = types.SimpleNamespace(entries=[entry, broken])

            deals = ScrapedDeal.fetch()

        self.assertEqual(len(deals), 1)
        self.assertEqual(deals[0].title, "Working deal")

    def test_scraped_deal_fetch_filters_non_new_conditions(self):
        self._install_model_stubs()
        from ingest.scraper import ScrapedDeal

        entries = [
            {
                "title": "New Portable Power Station",
                "summary": "<div class='snippet summary'>brand new retail deal</div>",
                "links": [{"href": "https://example.test/deal/1.html"}],
            },
            {
                "title": "Refurbished Laptop",
                "summary": "<div class='snippet summary'>factory refurbished</div>",
                "links": [{"href": "https://example.test/deal/2.html"}],
            },
            {
                "title": "Open-Box Headphones",
                "summary": "<div class='snippet summary'>open box item</div>",
                "links": [{"href": "https://example.test/deal/3.html"}],
            },
            {
                "title": "Used Tablet",
                "summary": "<div class='snippet summary'>pre-owned condition</div>",
                "links": [{"href": "https://example.test/deal/4.html"}],
            },
        ]

        def fake_get(url, timeout=None):
            return types.SimpleNamespace(
                content=b"<div class='content-section'>New retail product Features Full warranty</div>",
                raise_for_status=lambda: None,
            )

        with (
            patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "0"}),
            patch("ingest.scraper.FEEDS", ["https://example.test/rss"]),
            patch("ingest.scraper.feedparser.parse") as parse,
            patch("ingest.scraper.requests.get", side_effect=fake_get),
        ):
            parse.return_value = types.SimpleNamespace(entries=entries)

            deals = ScrapedDeal.fetch()

        self.assertEqual([deal.title for deal in deals], ["New Portable Power Station"])

    def test_scraped_deal_fetch_keeps_new_items_with_used_as_a_verb(self):
        self._install_model_stubs()
        from ingest.scraper import ScrapedDeal

        entry = {
            "title": "New USB-C Charger",
            "summary": "<div class='snippet summary'>can be used for phones and tablets</div>",
            "links": [{"href": "https://example.test/deal/1.html"}],
        }

        def fake_get(url, timeout=None):
            return types.SimpleNamespace(
                content=b"<div class='content-section'>This charger can be used for Apple and Android devices Features Full warranty</div>",
                raise_for_status=lambda: None,
            )

        with (
            patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "0"}),
            patch("ingest.scraper.FEEDS", ["https://example.test/rss"]),
            patch("ingest.scraper.feedparser.parse") as parse,
            patch("ingest.scraper.requests.get", side_effect=fake_get),
        ):
            parse.return_value = types.SimpleNamespace(entries=[entry])

            deals = ScrapedDeal.fetch()

        self.assertEqual([deal.title for deal in deals], ["New USB-C Charger"])

    def test_scraped_deal_extracts_list_price_from_detail_text(self):
        self._install_model_stubs()
        from ingest.scraper import ScrapedDeal

        entry = {
            "title": "New Monitor",
            "summary": "<div class='snippet summary'>Now $190. List price $299.99.</div>",
            "links": [{"href": "https://example.test/deal/1.html"}],
        }

        def fake_get(url, timeout=None):
            return types.SimpleNamespace(
                content=b"<div class='content-section'>Details List Price: $299.99 Features QHD display</div>",
                raise_for_status=lambda: None,
            )

        with (
            patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "0"}),
            patch("ingest.scraper.requests.get", side_effect=fake_get),
        ):
            deal = ScrapedDeal(entry)

        self.assertEqual(deal.list_price, 299.99)
        self.assertIn("List Price: $299.99", deal.describe())

    def test_extract_list_price_supports_dealnews_double_price(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = "Renogy 2000W Pure Sine Wave Inverter $167 $482 free shipping more"

        self.assertEqual(extract_list_price(text, deal_price=167.0), 482.0)

    def test_extract_list_price_prefers_dealnews_double_price_over_model_numbers(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = (
            "Aoostar Maco 6850H AMD Ryzen 7 Pro Mini PC $266 $861 free shipping more. "
            "Features AMD Ryzen 7 Pro 6850H 24GB LPDDR5 RAM. "
            "Related Offers Other Gaming Desktop $4,554 $9,738 free shipping."
        )

        self.assertEqual(extract_list_price(text, deal_price=266.0), 861.0)

    def test_extract_list_price_supports_promo_terms_between_dealnews_prices(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = "Beats Studio Pro Headphones $132 w/ Prime $200 free shipping more"

        self.assertEqual(extract_list_price(text, deal_price=132.0), 200.0)

    def test_extract_list_price_supports_rounded_display_price(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = "Energizer AAA Batteries $15 w/ Sub & Save $20 free shipping more"

        self.assertEqual(extract_list_price(text, deal_price=15.19), 20.0)

    def test_extract_list_price_supports_stacked_percentage_discounts(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = (
            "Wavlink DisplayLink Docking Station 50% off + Extra 20% off + "
            "Extra 6% off $150 free shipping. Price start at $56."
        )

        self.assertEqual(extract_list_price(text, deal_price=56.0), 150.0)

    def test_extract_list_price_ignores_related_offer_percentage_discounts(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = (
            "Wavlink DisplayLink Docking Station 50% off + Extra 20% off + "
            "Extra 6% off $150 free shipping. Price start at $56. "
            "Related Offers Laptop Screen Extender $144 $300. That's 52% off."
        )

        self.assertEqual(extract_list_price(text, deal_price=56.0), 150.0)

    def test_extract_list_price_supports_regular_price_of(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = "This inverter is $196 off its regular price of $482."

        self.assertEqual(extract_list_price(text, deal_price=167.0), 482.0)

    def test_extract_list_price_supports_from_its_list_price(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = "Avapow jump starter for $70, down $90 from its $160 list price."

        self.assertEqual(extract_list_price(text, deal_price=70.0), 160.0)

    def test_extract_list_price_avoids_related_offer_without_deal_price(self):
        self._install_model_stubs()
        from ingest.list_price import extract_list_price

        text = (
            "Target deal $70 free shipping. Related Offers: "
            "Other product $19 $99 free shipping more."
        )

        self.assertIsNone(extract_list_price(text, deal_price=70.0))

    def _install_deal_framework_stubs(self):
        if "app.orchestrator" in sys.modules:
            return

        self._install_model_stubs()

        agent_mcp = types.ModuleType("app.mcp_client")
        agent_mcp.run_sync = lambda memory: (memory, None)
        sys.modules.setdefault("app.mcp_client", agent_mcp)

        chromadb = types.ModuleType("chromadb")
        chromadb.PersistentClient = lambda path: None
        sys.modules.setdefault("chromadb", chromadb)

        sklearn = types.ModuleType("sklearn")
        manifold = types.ModuleType("sklearn.manifold")
        manifold.TSNE = object
        sys.modules.setdefault("sklearn", sklearn)
        sys.modules.setdefault("sklearn.manifold", manifold)

        numpy = types.ModuleType("numpy")
        numpy.array = lambda value: value
        sys.modules.setdefault("numpy", numpy)

    def _install_model_stubs(self):
        feedparser = types.ModuleType("feedparser")
        feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
        sys.modules.setdefault("feedparser", feedparser)


if __name__ == "__main__":
    unittest.main()

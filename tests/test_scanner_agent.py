import sys
import types
import unittest
from types import SimpleNamespace


feedparser = types.ModuleType("feedparser")
feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
sys.modules.setdefault("feedparser", feedparser)

from agents.scanner_agent import ScannerAgent
from domain.deal import Deal, DealSelection


class _FakeScraped:
    def __init__(self, url, price, list_price):
        self.url = url
        self.price = price
        self.list_price = list_price

    def describe(self):
        return f"Title: Example\nList Price: {self.list_price}\nURL: {self.url}"


class ScannerAttachTests(unittest.TestCase):
    def test_list_price_reattached_by_id_when_llm_drops_it(self):
        agent = ScannerAgent.__new__(ScannerAgent)  # skip OpenAI() in __init__

        # The model selects the deal but omits list_price and echoes a different slug.
        llm_deal = Deal(
            product_description="Example",
            price=150.0,
            list_price=None,
            url="https://www.dealnews.com/products/Different-Slug/480839.html",
        )
        result = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=DealSelection(deals=[llm_deal])))]
        )
        agent.openai = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kw: result))
        )
        scraped = [
            _FakeScraped(
                "https://www.dealnews.com/products/Original-Slug/480839.html?iref=rss",
                price=150.0,
                list_price=250.0,
            )
        ]
        agent.fetch_deals = lambda memory: scraped

        selection = agent.scan(memory=[])

        # Scraped list_price wins over the model's None, matched on product id 480839.
        self.assertEqual(selection.deals[0].list_price, 250.0)


if __name__ == "__main__":
    unittest.main()

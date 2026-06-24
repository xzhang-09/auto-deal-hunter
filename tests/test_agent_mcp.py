import sys
import types
import unittest


feedparser = types.ModuleType("feedparser")
feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
sys.modules.setdefault("feedparser", feedparser)

# Other test modules (e.g. test_ui_helpers) install a lightweight ``app.agent_mcp`` stub
# exposing only ``run_sync`` to dodge its heavy MCP/OpenAI imports. This test exercises the
# real ``opportunity_from_notify_args``, so drop any such stub and import the genuine module,
# keeping the test independent of collection order.
sys.modules.pop("app.agent_mcp", None)
from app import agent_mcp


class AgentMcpTests(unittest.TestCase):
    def test_max_agent_steps_is_bounded(self):
        self.assertEqual(agent_mcp.MAX_AGENT_STEPS, 8)

    def test_opportunity_from_notify_args_keeps_scanned_list_price(self):
        args = {
            "description": "Example monitor",
            "deal_price": 190.0,
            "estimated_true_value": 299.99,
            "url": "https://example.test/deal/1.html",
        }
        scanned = {
            args["url"]: {
                "product_description": "Example monitor",
                "price": 190.0,
                "list_price": 249.99,
                "url": args["url"],
            }
        }

        opportunity = agent_mcp.opportunity_from_notify_args(args, scanned)

        self.assertEqual(opportunity.deal.list_price, 249.99)
        # estimate 299.99 > list 249.99, so the value is capped at list before the discount:
        # 249.99 - 190 = 59.99, not the raw 299.99 - 190.
        self.assertAlmostEqual(opportunity.discount, 59.99)

    def test_parse_estimate_reads_dollar_value(self):
        self.assertEqual(
            agent_mcp._parse_estimate("The estimated true value of this product is $1,299.50"),
            1299.50,
        )
        self.assertIsNone(agent_mcp._parse_estimate("no price here"))

    def test_candidate_from_estimate_pairs_by_description(self):
        scanned = {
            "https://x.test/1.html": {
                "product_description": "A nice monitor",
                "price": 190.0,
                "list_price": 249.99,
                "url": "https://x.test/1.html",
            }
        }
        candidate = agent_mcp.candidate_from_estimate("A nice monitor", 300.0, scanned)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.deal.list_price, 249.99)
        # estimate 300 capped at list 249.99 -> discount 59.99
        self.assertAlmostEqual(candidate.discount, 59.99)

    def test_candidate_from_estimate_returns_none_when_unmatched(self):
        scanned = {
            "https://x.test/1.html": {
                "product_description": "A nice monitor",
                "price": 190.0,
                "url": "https://x.test/1.html",
            }
        }
        self.assertIsNone(agent_mcp.candidate_from_estimate("paraphrased text", 300.0, scanned))

    def test_opportunity_recovers_list_price_by_id_when_url_differs(self):
        args = {
            "description": "Example monitor",
            "deal_price": 190.0,
            "estimated_true_value": 299.99,
            "url": "https://example.test/products/New-Slug/1.html",
        }
        # Scanned under a different slug/query but the same product id (1).
        scanned = {
            "https://example.test/products/Old-Slug/1.html?iref=rss": {
                "product_description": "Example monitor",
                "price": 190.0,
                "list_price": 249.99,
                "url": "https://example.test/products/Old-Slug/1.html?iref=rss",
            }
        }

        opportunity = agent_mcp.opportunity_from_notify_args(args, scanned)

        self.assertEqual(opportunity.deal.list_price, 249.99)


if __name__ == "__main__":
    unittest.main()

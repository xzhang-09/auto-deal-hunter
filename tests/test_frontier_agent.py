import unittest
import sys
import types


sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = lambda *args, **kwargs: None
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from agents.frontier_agent import FrontierAgent


class FrontierAgentParsingTests(unittest.TestCase):
    def test_get_price_accepts_plain_price(self):
        self.assertEqual(FrontierAgent.get_price("$1,299.99"), 1299.99)

    def test_get_price_accepts_json_price(self):
        self.assertEqual(FrontierAgent.get_price('{"price": 249.5}'), 249.5)

    def test_get_price_rejects_ranges(self):
        with self.assertRaises(ValueError):
            FrontierAgent.get_price("$90 to $120")

    def test_get_price_rejects_missing_price(self):
        with self.assertRaises(ValueError):
            FrontierAgent.get_price("unknown")


if __name__ == "__main__":
    unittest.main()

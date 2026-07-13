import importlib
import sys
import types
import unittest


class McpServerCacheTests(unittest.TestCase):
    def test_get_agents_reuses_cached_instances(self):
        feedparser = types.ModuleType("feedparser")
        feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
        sys.modules.setdefault("feedparser", feedparser)

        sentence_transformers = types.ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = lambda *args, **kwargs: None
        sys.modules.setdefault("sentence_transformers", sentence_transformers)

        chromadb = types.ModuleType("chromadb")
        chromadb.PersistentClient = lambda path: types.SimpleNamespace(
            get_or_create_collection=lambda name: object()
        )
        sys.modules["chromadb"] = chromadb

        module = importlib.import_module("auto_deal_hunter.app.mcp_server")
        module._AGENTS_CACHE = None
        module.ScannerAgent = lambda: object()
        module.PricerAgent = lambda collection: object()
        module.MessagingAgent = lambda: object()

        first = module._get_agents()
        second = module._get_agents()

        self.assertIs(first, second)
        self.assertIs(first["estimator"], second["estimator"])


class NotifyConfidenceGateTests(unittest.TestCase):
    def _load(self):
        feedparser = types.ModuleType("feedparser")
        feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
        sys.modules.setdefault("feedparser", feedparser)
        st = types.ModuleType("sentence_transformers")
        st.SentenceTransformer = lambda *a, **k: None
        sys.modules.setdefault("sentence_transformers", st)
        chromadb = types.ModuleType("chromadb")
        chromadb.PersistentClient = lambda path: types.SimpleNamespace(
            get_or_create_collection=lambda name: object()
        )
        sys.modules["chromadb"] = chromadb

        module = importlib.import_module("auto_deal_hunter.app.mcp_server")
        sent = []
        original_get_agents = module._get_agents
        original_threshold = module.RAG_MIN_CONFIDENCE
        module._get_agents = lambda: {
            "messenger": types.SimpleNamespace(notify=lambda *a, **k: sent.append(a))
        }
        module.RAG_MIN_CONFIDENCE = 0.15
        module._CONFIDENCE_BY_ID.clear()

        def restore():
            module._get_agents = original_get_agents
            module.RAG_MIN_CONFIDENCE = original_threshold
            module._CONFIDENCE_BY_ID.clear()

        self.addCleanup(restore)
        return module, sent

    def test_low_confidence_withholds_push(self):
        module, sent = self._load()
        url = "https://x.test/1.html"
        module._CONFIDENCE_BY_ID[module.deal_id(url)] = 0.05

        result = module.notify_deal("desc", 50.0, 100.0, url)

        self.assertIn("withheld", result.lower())
        self.assertEqual(sent, [])  # no push sent

    def test_high_confidence_sends_push(self):
        module, sent = self._load()
        url = "https://x.test/1.html"
        module._CONFIDENCE_BY_ID[module.deal_id(url)] = 0.9

        result = module.notify_deal("desc", 50.0, 100.0, url)

        self.assertIn("sent", result.lower())
        self.assertEqual(len(sent), 1)

    def test_list_price_is_forwarded_to_messenger(self):
        module, sent = self._load()

        module.notify_deal("desc", 50.0, 100.0, "https://x.test/1.html", 80.0)

        self.assertEqual(sent[0][4], 80.0)

    def test_unknown_confidence_is_not_gated(self):
        # A deal that was never estimated (no confidence recorded) is pushed, not suppressed.
        module, sent = self._load()

        result = module.notify_deal("desc", 50.0, 100.0, "https://x.test/never/9.html")

        self.assertIn("sent", result.lower())
        self.assertEqual(len(sent), 1)


if __name__ == "__main__":
    unittest.main()

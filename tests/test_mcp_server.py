import importlib
import sys
import types
import unittest


class McpServerCacheTests(unittest.TestCase):
    def test_get_agents_reuses_cached_instances(self):
        feedparser = types.ModuleType("feedparser")
        feedparser.parse = lambda url: types.SimpleNamespace(entries=[])
        sys.modules.setdefault("feedparser", feedparser)

        litellm = types.ModuleType("litellm")
        litellm.completion = lambda *args, **kwargs: None
        sys.modules.setdefault("litellm", litellm)

        sentence_transformers = types.ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = lambda *args, **kwargs: None
        sys.modules.setdefault("sentence_transformers", sentence_transformers)

        chromadb = types.ModuleType("chromadb")
        chromadb.PersistentClient = lambda path: types.SimpleNamespace(
            get_or_create_collection=lambda name: object()
        )
        sys.modules["chromadb"] = chromadb

        module = importlib.import_module("app.mcp_server")
        module._AGENTS_CACHE = None
        module.ScannerAgent = lambda: object()
        module.FrontierAgent = lambda collection: object()
        module.MessagingAgent = lambda: object()

        first = module._get_agents()
        second = module._get_agents()

        self.assertIs(first, second)
        self.assertIs(first["estimator"], second["estimator"])


if __name__ == "__main__":
    unittest.main()

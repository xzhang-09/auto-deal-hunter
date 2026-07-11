import sys
import types
import unittest


sentence_transformers = sys.modules.setdefault(
    "sentence_transformers", types.ModuleType("sentence_transformers")
)


class FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        return [len(candidate) for _, candidate in pairs]


sentence_transformers.CrossEncoder = FakeCrossEncoder
sentence_transformers.SentenceTransformer = lambda *args, **kwargs: None

from core.reranker import CrossEncoderReranker, LLMReranker, NoopReranker


class RerankerTests(unittest.TestCase):
    def test_noop_preserves_order(self):
        reranker = NoopReranker()

        self.assertEqual(reranker.rerank("query", ["b", "a"]), [0, 1])

    def test_cross_encoder_sorts_by_score_descending(self):
        reranker = CrossEncoderReranker()

        self.assertEqual(reranker.rerank("query", ["short", "much longer", "mid"]), [1, 0, 2])

    def test_llm_reranker_accepts_valid_structured_order(self):
        result = types.SimpleNamespace(ranked_indices=[2, 0, 1])
        message = types.SimpleNamespace(parsed=result)
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)
        completions = types.SimpleNamespace(parse=lambda **kwargs: response)
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        reranker = LLMReranker(client=client, model="test-model")

        self.assertEqual(reranker.rerank("query", ["a", "b", "c"]), [2, 0, 1])

    def test_llm_reranker_falls_back_on_illegal_index(self):
        result = types.SimpleNamespace(ranked_indices=[0, 4, 1])
        message = types.SimpleNamespace(parsed=result)
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)
        completions = types.SimpleNamespace(parse=lambda **kwargs: response)
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        reranker = LLMReranker(client=client, model="test-model")

        self.assertEqual(reranker.rerank("query", ["a", "b", "c"]), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

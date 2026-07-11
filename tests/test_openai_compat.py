import types
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from infra import openai_compat
from infra.usage import UsageTracker


class ExampleResult(BaseModel):
    value: int


class OpenAICompatTests(unittest.TestCase):
    def setUp(self):
        from infra import usage

        usage.TRACKER.reset()
        self.addCleanup(usage.TRACKER.reset)

    def test_parse_structured_uses_responses_when_available(self):
        parsed = ExampleResult(value=3)
        content = types.SimpleNamespace(parsed=parsed)
        output = types.SimpleNamespace(content=[content])
        response = types.SimpleNamespace(
            output=[output],
            usage=types.SimpleNamespace(input_tokens=10, output_tokens=2),
        )
        client = types.SimpleNamespace(
            responses=types.SimpleNamespace(parse=lambda **kwargs: response)
        )

        with patch.object(openai_compat, "OPENAI_API_STYLE", "responses"):
            result = openai_compat.parse_structured(
                client,
                model="test-model",
                user_prompt="prompt",
                text_format=ExampleResult,
            )

        self.assertEqual(result.value, 3)

    def test_generate_text_uses_responses_output_text(self):
        response = types.SimpleNamespace(output_text="hello", usage=None)
        client = types.SimpleNamespace(
            responses=types.SimpleNamespace(create=lambda **kwargs: response)
        )

        with patch.object(openai_compat, "OPENAI_API_STYLE", "responses"):
            text = openai_compat.generate_text(client, model="test-model", user_prompt="prompt")

        self.assertEqual(text, "hello")

    def test_chat_fallback_when_responses_unavailable(self):
        parsed = ExampleResult(value=7)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(parsed=parsed))],
            usage=None,
        )
        completions = types.SimpleNamespace(parse=lambda **kwargs: response)
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

        with patch.object(openai_compat, "OPENAI_API_STYLE", "responses"):
            result = openai_compat.parse_structured(
                client,
                model="test-model",
                user_prompt="prompt",
                text_format=ExampleResult,
            )

        self.assertEqual(result.value, 7)

    def test_usage_tracker_accepts_responses_usage_fields(self):
        tracker = UsageTracker()

        tracker.record("gpt-4o-mini", types.SimpleNamespace(input_tokens=12, output_tokens=4))

        self.assertEqual(tracker.prompt_tokens, 12)
        self.assertEqual(tracker.completion_tokens, 4)
        self.assertEqual(tracker.calls, 1)


if __name__ == "__main__":
    unittest.main()

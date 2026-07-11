import types
import unittest

from domain.deal import Deal, Opportunity
from evaluation.judge import MessageJudge, ScanJudge, corrupted_variants, MessageVerdict


class MessageJudgeTests(unittest.TestCase):
    def test_judge_returns_structured_verdict(self):
        parsed = MessageVerdict(faithful=True, issues=[], score=5)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(parsed=parsed))],
            usage=None,
        )
        completions = types.SimpleNamespace(parse=lambda **kwargs: response)
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        judge = MessageJudge(client=client, model="test-model")

        verdict = judge.judge(self._opportunity(), "Great deal at $50.")

        self.assertTrue(verdict.faithful)
        self.assertEqual(verdict.score, 5)

    def test_judge_json_fallback(self):
        def parse(**kwargs):
            raise AttributeError("parse unavailable")

        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"faithful": false, "issues": ["overstates savings"], "score": 2}'
                    )
                )
            ],
            usage=None,
        )
        completions = types.SimpleNamespace(parse=parse, create=lambda **kwargs: response)
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        judge = MessageJudge(client=client, model="test-model")

        verdict = judge.judge(self._opportunity(), "Save $500 on this $50 item.")

        self.assertFalse(verdict.faithful)
        self.assertEqual(verdict.issues, ["overstates savings"])
        self.assertEqual(verdict.score, 2)

    def test_prompt_includes_core_facts(self):
        prompt = MessageJudge._prompt(self._opportunity(), "message")

        self.assertIn("Example product", prompt)
        self.assertIn("$50.00", prompt)
        self.assertIn("$100.00", prompt)
        self.assertIn("Capped savings", prompt)

    def test_corrupted_variants_rescale_first_price(self):
        message = "Grab it for $50.00 (valued at $100.00)!"
        variants = corrupted_variants(message)
        self.assertEqual(
            variants["halved_price"], "Grab it for $25.00 (valued at $100.00)!"
        )
        self.assertEqual(
            variants["doubled_price"], "Grab it for $100.00 (valued at $100.00)!"
        )

    def test_corrupted_variants_handle_commas_and_no_price(self):
        variants = corrupted_variants("MacBook for $1,299.99 today")
        self.assertEqual(variants["halved_price"], "MacBook for $650.00 today")
        # No dollar amount: price corruptions are skipped, invented fact still applies.
        no_price = corrupted_variants("A great deal on headphones")
        self.assertNotIn("halved_price", no_price)
        self.assertNotIn("doubled_price", no_price)
        self.assertIn("free 5-year extended warranty", no_price["invented_fact"])

    def test_invented_fact_appends_unavailable_details(self):
        variants = corrupted_variants("Speaker for $80.")
        self.assertTrue(variants["invented_fact"].startswith("Speaker for $80."))
        self.assertIn("$50 gift card", variants["invented_fact"])

    def test_scan_judge_prompt_includes_listing_and_output(self):
        deal = Deal(
            product_description="Wireless mouse with USB receiver",
            price=12.5,
            url="https://example.test/mouse",
        )
        prompt = ScanJudge._prompt("Title: Wireless Mouse\nDetails: $12.50", deal)
        self.assertIn("Title: Wireless Mouse", prompt)
        self.assertIn("Wireless mouse with USB receiver", prompt)
        self.assertIn("$12.50", prompt)
        self.assertNotIn("PER UNIT", prompt)

    def test_scan_judge_prompt_notes_per_unit_rebasing(self):
        deal = Deal(
            product_description="AA batteries",
            price=0.5,
            url="https://example.test/batteries",
            quantity=36,
        )
        prompt = ScanJudge._prompt("Title: AA 36-pack for $18", deal)
        self.assertIn("Pack size: 36", prompt)
        self.assertIn("PER UNIT", prompt)

    def _opportunity(self):
        return Opportunity(
            deal=Deal(
                product_description="Example product",
                price=50.0,
                list_price=120.0,
                url="https://example.test/deal",
            ),
            estimate=100.0,
        )


if __name__ == "__main__":
    unittest.main()

import unittest
import json
from unittest.mock import Mock, patch


class MessagingAgentTests(unittest.TestCase):
    @patch("auto_deal_hunter.agents.messaging_agent.OpenAI")
    @patch("auto_deal_hunter.agents.messaging_agent.requests.post")
    def test_push_uses_telegram_when_configured(self, post, _openai):
        from auto_deal_hunter.agents.messaging_agent import MessagingAgent

        response = Mock()
        response.raise_for_status.return_value = None
        post.return_value = response

        with patch.dict(
            "os.environ",
            {
                "TELEGRAM_BOT_TOKEN": "bot-token",
                "TELEGRAM_CHAT_ID": "123456",
                "PUSHOVER_USER": "pushover-user",
                "PUSHOVER_TOKEN": "pushover-token",
            },
            clear=False,
        ):
            MessagingAgent().push("deal text")

        post.assert_called_once_with(
            "https://api.telegram.org/botbot-token/sendMessage",
            data={"chat_id": "123456", "text": "deal text", "disable_web_page_preview": False},
            timeout=10,
        )

    @patch("auto_deal_hunter.agents.messaging_agent.OpenAI")
    @patch("auto_deal_hunter.agents.messaging_agent.requests.post")
    def test_notify_includes_prices_and_feedback_buttons(self, post, _openai):
        from auto_deal_hunter.agents.messaging_agent import MessagingAgent

        response = Mock()
        response.raise_for_status.return_value = None
        post.return_value = response

        with patch.dict(
            "os.environ",
            {"TELEGRAM_BOT_TOKEN": "bot-token", "TELEGRAM_CHAT_ID": "123456"},
            clear=False,
        ), patch("auto_deal_hunter.agents.messaging_agent.TELEGRAM_FEEDBACK_ENABLED", True):
            agent = MessagingAgent()
            agent.craft_message = Mock(return_value="A concise deal summary.")
            agent.notify("Product", 22.99, 24.99, "https://dealnews.test/504998.html", 29.99)

        payload = post.call_args.kwargs["data"]
        self.assertIn("Deal price: $22.99", payload["text"])
        self.assertIn("List price: $29.99", payload["text"])
        self.assertIn("Estimated value: ~$25", payload["text"])
        # Estimate (24.99) below list (29.99): cap not active, so savings are the
        # estimate-derived approximation and carry no misleading "capped" note.
        self.assertIn("Savings: ~$2", payload["text"])
        self.assertNotIn("capped", payload["text"])
        self.assertTrue(payload["text"].endswith("https://dealnews.test/504998.html"))
        buttons = json.loads(payload["reply_markup"])["inline_keyboard"][0]
        self.assertEqual(buttons[0]["callback_data"], "fb:g:504998")
        self.assertEqual(buttons[1]["callback_data"], "fb:b:504998")

    @patch("auto_deal_hunter.agents.messaging_agent.OpenAI")
    @patch("auto_deal_hunter.agents.messaging_agent.requests.post")
    def test_feedback_buttons_are_omitted_when_listener_is_disabled(self, post, _openai):
        from auto_deal_hunter.agents.messaging_agent import MessagingAgent

        post.return_value = Mock(raise_for_status=Mock())
        with patch.dict(
            "os.environ",
            {"TELEGRAM_BOT_TOKEN": "bot-token", "TELEGRAM_CHAT_ID": "123456"},
            clear=False,
        ), patch("auto_deal_hunter.agents.messaging_agent.TELEGRAM_FEEDBACK_ENABLED", False):
            MessagingAgent().push("deal", "504998")

        self.assertNotIn("reply_markup", post.call_args.kwargs["data"])

    def test_message_omits_unknown_list_price(self):
        from auto_deal_hunter.agents.messaging_agent import MessagingAgent

        message = MessagingAgent._message_with_prices(
            "Summary", 7.0, 15.0, "https://example.test/deal", None
        )

        self.assertNotIn("List price", message)
        self.assertNotIn("capped", message)
        self.assertIn("Savings: ~$8", message)

    def test_savings_are_capped_when_estimate_exceeds_list_price(self):
        from auto_deal_hunter.agents.messaging_agent import MessagingAgent

        # Estimate above list but below the mismatch ratio: shown, savings capped at list.
        message = MessagingAgent._message_with_prices(
            "Summary", 350.0, 500.0, "https://example.test/deal", 448.0
        )

        self.assertIn("Estimated value: ~$500", message)
        self.assertIn("Savings (capped at list): $98.00", message)

    def test_mismatch_estimate_is_omitted_from_message(self):
        from auto_deal_hunter.agents.messaging_agent import MessagingAgent

        # Estimate at 3x list price (comparables mismatch): no estimate line, mirroring the
        # dashboard's ⚠️ n/a; capped savings remain because they are real seller prices.
        message = MessagingAgent._message_with_prices(
            "Summary", 350.0, 1400.0, "https://example.test/deal", 448.0
        )

        self.assertNotIn("Estimated value", message)
        self.assertIn("Savings (capped at list): $98.00", message)

    def test_multipack_message_shows_pack_level_prices(self):
        from auto_deal_hunter.agents.messaging_agent import MessagingAgent

        # The battery case: per-unit inputs, 48-pack. Mismatch (20.9 > 2x 0.8125) drops the
        # estimate; every remaining figure is pack-level, matching the dashboard row.
        message = MessagingAgent._message_with_prices(
            "Summary", 0.5, 20.9046908, "https://example.test/deal", 0.8125, 48
        )

        self.assertIn("Deal price: $24.00", message)
        self.assertIn("List price: $39.00", message)
        self.assertNotIn("Estimated value", message)
        self.assertIn("Savings (capped at list): $15.00", message)



if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import Mock, patch


class MessagingAgentTests(unittest.TestCase):
    @patch("agents.messaging_agent.OpenAI")
    @patch("agents.messaging_agent.requests.post")
    def test_push_uses_telegram_when_configured(self, post, _openai):
        from agents.messaging_agent import MessagingAgent

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


if __name__ == "__main__":
    unittest.main()

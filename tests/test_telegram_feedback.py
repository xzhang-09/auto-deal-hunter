import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from core.opportunity_store import OpportunityStore
from core.source_ids import deal_id
from domain.deal import Deal, Opportunity
from infra.telegram_feedback import TelegramFeedbackPoller


class TelegramFeedbackPollerTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.store = OpportunityStore(Path(self.tmpdir.name) / "deals.sqlite")
        self.opportunity = Opportunity(
            deal=Deal(
                product_description="Product",
                price=20.0,
                list_price=40.0,
                url="https://dealnews.test/item/12345.html",
            ),
            estimate=35.0,
        )
        self.store.append(self.opportunity)
        self.poller = TelegramFeedbackPoller("token", "777", self.store)

    @staticmethod
    def _update(data="fb:g:12345", chat_id=777):
        return {
            "update_id": 1,
            "callback_query": {
                "id": "callback-1",
                "data": data,
                "message": {"message_id": 9, "chat": {"id": chat_id}},
            },
        }

    @patch("infra.telegram_feedback.requests.post")
    def test_good_feedback_is_saved_and_message_is_updated(self, post):
        post.return_value = Mock(raise_for_status=Mock())

        self.poller.process_update(self._update())

        self.assertEqual(
            self.store.feedback_map()[deal_id(self.opportunity.deal.url)], "good_deal"
        )
        self.assertEqual(post.call_count, 2)
        self.assertIn("answerCallbackQuery", post.call_args_list[0].args[0])
        self.assertIn("editMessageReplyMarkup", post.call_args_list[1].args[0])
        markup = json.loads(post.call_args_list[1].kwargs["data"]["reply_markup"])
        self.assertEqual(markup["inline_keyboard"][0][0]["text"], "✅ Good deal")

    @patch("infra.telegram_feedback.requests.post")
    def test_bad_feedback_can_replace_good_feedback(self, post):
        post.return_value = Mock(raise_for_status=Mock())
        self.poller.process_update(self._update())

        self.poller.process_update(self._update("fb:b:12345"))

        self.assertEqual(
            self.store.feedback_map()[deal_id(self.opportunity.deal.url)], "bad_deal"
        )

    @patch("infra.telegram_feedback.requests.post")
    def test_unauthorized_chat_cannot_write_feedback(self, post):
        post.return_value = Mock(raise_for_status=Mock())

        self.poller.process_update(self._update(chat_id=999))

        self.assertEqual(self.store.feedback_map(), {})
        self.assertEqual(post.call_count, 1)
        self.assertTrue(post.call_args.kwargs["data"]["show_alert"])

    @patch("infra.telegram_feedback.requests.post")
    def test_unknown_deal_is_not_recorded(self, post):
        post.return_value = Mock(raise_for_status=Mock())

        self.poller.process_update(self._update("fb:g:missing"))

        self.assertEqual(self.store.feedback_map(), {})
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()

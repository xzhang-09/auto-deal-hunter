import json
import logging
import threading

import requests

from core.opportunity_store import OpportunityStore


class TelegramFeedbackPoller:
    def __init__(
        self,
        token: str,
        chat_id: str,
        store: OpportunityStore,
        poll_timeout: int = 25,
    ):
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.chat_id = str(chat_id)
        self.store = store
        self.poll_timeout = poll_timeout
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="telegram-feedback", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        offset = None
        while not self._stop.is_set():
            params = {
                "timeout": self.poll_timeout,
                "allowed_updates": json.dumps(["callback_query"]),
            }
            if offset is not None:
                params["offset"] = offset
            try:
                response = requests.get(
                    f"{self.base_url}/getUpdates",
                    params=params,
                    timeout=self.poll_timeout + 5,
                )
                response.raise_for_status()
                updates = response.json().get("result", [])
                for update in updates:
                    offset = max(offset or 0, update["update_id"] + 1)
                    self.process_update(update)
            except (requests.RequestException, ValueError, KeyError) as exc:
                logging.warning("Telegram feedback polling failed: %s", exc)
                self._stop.wait(2)

    def process_update(self, update: dict) -> None:
        callback = update.get("callback_query")
        if not callback:
            return
        callback_id = callback.get("id", "")
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        if str(chat.get("id", "")) != self.chat_id:
            self._answer(callback_id, "Unauthorized chat", alert=True)
            return

        parts = (callback.get("data") or "").split(":", 2)
        if len(parts) != 3 or parts[0] != "fb" or parts[1] not in {"g", "b"}:
            self._answer(callback_id, "Invalid feedback", alert=True)
            return

        label = "good_deal" if parts[1] == "g" else "bad_deal"
        if not self.store.mark_feedback_by_id(parts[2], label):
            self._answer(callback_id, "Deal is no longer available", alert=True)
            return

        self._answer(callback_id, "Feedback saved")
        self._show_selection(chat["id"], message.get("message_id"), parts[2], label)

    def _answer(self, callback_id: str, text: str, alert: bool = False) -> None:
        if not callback_id:
            return
        self._post(
            "answerCallbackQuery",
            {"callback_query_id": callback_id, "text": text, "show_alert": alert},
        )

    def _show_selection(
        self, chat_id: int | str, message_id: int | None, dedup_id: str, label: str
    ) -> None:
        if message_id is None:
            return
        markup = {
            "inline_keyboard": [
                [
                    {
                        "text": "✅ Good deal" if label == "good_deal" else "👍 Good deal",
                        "callback_data": f"fb:g:{dedup_id}",
                    },
                    {
                        "text": "✅ Bad deal" if label == "bad_deal" else "👎 Bad deal",
                        "callback_data": f"fb:b:{dedup_id}",
                    },
                ]
            ]
        }
        self._post(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": json.dumps(markup),
            },
        )

    def _post(self, method: str, data: dict) -> None:
        try:
            response = requests.post(f"{self.base_url}/{method}", data=data, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            logging.warning("Telegram feedback response failed: %s", exc)

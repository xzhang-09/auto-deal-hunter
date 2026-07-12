import os
import json
from agents.agent import Agent
from core.identity_policy import display_description
from infra.config import (
    ESTIMATE_MISMATCH_RATIO,
    LLM_MAX_RETRIES,
    MESSAGING_MODEL,
    TELEGRAM_FEEDBACK_ENABLED,
)
from infra.openai_compat import generate_text
from openai import OpenAI
from domain.deal import Opportunity
import requests

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
TELEGRAM_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = MESSAGING_MODEL

    def __init__(self):
        self.log("Initializing")
        self.client = OpenAI(max_retries=LLM_MAX_RETRIES)
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.telegram_feedback_enabled = TELEGRAM_FEEDBACK_ENABLED
        self.pushover_user = os.getenv("PUSHOVER_USER", "")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "")
        self.log("Ready")

    def push(self, text: str, callback_deal_id: str | None = None):
        self.log("Sending push notification")
        if self.telegram_token and self.telegram_chat_id:
            self._push_telegram(text, callback_deal_id)
        elif self.pushover_user and self.pushover_token:
            self._push_pushover(text)
        else:
            self.log("Push notifications not configured - logging instead")
            self.log(text[:200])

    def _push_telegram(self, text: str, callback_deal_id: str | None = None):
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "disable_web_page_preview": False,
        }
        if callback_deal_id and self.telegram_feedback_enabled:
            payload["reply_markup"] = json.dumps(
                {
                    "inline_keyboard": [
                        [
                            {
                                "text": "👍 Good deal",
                                "callback_data": f"fb:g:{callback_deal_id}",
                            },
                            {
                                "text": "👎 Bad deal",
                                "callback_data": f"fb:b:{callback_deal_id}",
                            },
                        ]
                    ]
                }
            )
        try:
            response = requests.post(
                TELEGRAM_URL_TEMPLATE.format(token=self.telegram_token),
                data=payload,
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            self.log(f"Telegram notification failed: {exc}")

    def _push_pushover(self, text: str):
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": text,
            "sound": "cashregister",
        }
        try:
            response = requests.post(PUSHOVER_URL, data=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            self.log(f"Pushover notification failed: {exc}")

    def alert(self, opportunity: Opportunity):
        quantity = opportunity.deal.quantity
        text = self._message_with_prices(
            display_description(opportunity.deal.product_description, quantity),
            opportunity.deal.price,
            opportunity.estimate,
            opportunity.deal.url,
            opportunity.deal.list_price,
            quantity,
        )
        from core.source_ids import deal_id

        self.push(text, deal_id(opportunity.deal.url))

    def craft_message(
        self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        prompt = (
            "Summarize this deal in 2-3 sentences for a push notification.\n"
            f"Item: {description}\nPrice: {deal_price}\nEst. value: {estimated_true_value}\n"
            "Respond only with the message."
        )
        return generate_text(self.client, model=self.MODEL, user_prompt=prompt)

    def notify(
        self,
        description: str,
        deal_price: float,
        estimated_true_value: float,
        url: str,
        list_price: float | None = None,
        quantity: int = 1,
    ):
        self.log("Crafting message with LLM")
        # The message LLM sees the same pack-level basis the reader will: pack prices and a
        # plain "(N-pack)" description, not the internal per-unit values and pricing note.
        text = self.craft_message(
            display_description(description, quantity),
            deal_price * quantity,
            estimated_true_value * quantity,
        )
        from core.source_ids import deal_id

        message = self._message_with_prices(
            text, deal_price, estimated_true_value, url, list_price, quantity
        )
        self.push(message, deal_id(url))

    @staticmethod
    def _message_with_prices(
        text: str,
        deal_price: float,
        estimated_true_value: float,
        url: str,
        list_price: float | None = None,
        quantity: int = 1,
    ) -> str:
        # Inputs are per-unit for multipacks (quantity > 1); show the pack-level figures the
        # reader actually pays, mirroring the dashboard table.
        pack_price = deal_price * quantity
        pack_list = list_price * quantity if list_price is not None else None
        prices = [f"Deal price: ${pack_price:.2f}"]
        if pack_list is not None:
            prices.append(f"List price: ${pack_list:.2f}")
        # A comparables-mismatch estimate (a multiple of list price) is untrustworthy: omit it
        # rather than push a nonsense figure, mirroring the dashboard's "⚠️ n/a".
        mismatch = (
            list_price is not None
            and estimated_true_value > ESTIMATE_MISMATCH_RATIO * list_price
        )
        if not mismatch:
            # The estimate is a model output with tens-of-dollars uncertainty; cents would be
            # false precision. Deal/list prices are real seller prices, so they keep cents.
            prices.append(f"Estimated value: ~${estimated_true_value * quantity:,.0f}")
        # State our own defensible savings (list-price-capped, mirroring
        # Opportunity.effective_value): otherwise the Telegram link preview's uncapped
        # "$X under list price" claim is the only savings figure on screen. Precision and
        # label follow which bound is active: capped savings are the difference of two real
        # seller prices (exact, cents kept, "capped" noted); uncapped savings inherit the
        # estimate's uncertainty (rounded, ~ prefix, no misleading "capped" note).
        if list_price is not None and estimated_true_value > list_price:
            prices.append(
                f"Savings (capped at list): ${(list_price - deal_price) * quantity:.2f}"
            )
        else:
            prices.append(
                f"Savings: ~${(estimated_true_value - deal_price) * quantity:,.0f}"
            )
        return f"{text.strip()}\n\n" + "\n".join(prices) + f"\n\n{url}"

import os
from agents.agent import Agent
from infra.config import LLM_MAX_RETRIES, LLM_MODEL, LLM_TEMPERATURE
from infra import usage
from openai import OpenAI
from domain.deal import Opportunity
import requests

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
TELEGRAM_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = LLM_MODEL

    def __init__(self):
        self.log("Initializing")
        self.client = OpenAI(max_retries=LLM_MAX_RETRIES)
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.pushover_user = os.getenv("PUSHOVER_USER", "")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "")
        self.log("Ready")

    def push(self, text: str):
        self.log("Sending push notification")
        if self.telegram_token and self.telegram_chat_id:
            self._push_telegram(text)
        elif self.pushover_user and self.pushover_token:
            self._push_pushover(text)
        else:
            self.log("Push notifications not configured - logging instead")
            self.log(text[:200])

    def _push_telegram(self, text: str):
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "disable_web_page_preview": False,
        }
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
        text = (
            f"Deal! Price=${opportunity.deal.price:.2f}, "
            f"Estimate=${opportunity.estimate:.2f}, "
            f"Discount=${opportunity.discount:.2f}: "
            f"{opportunity.deal.product_description[:50]}... {opportunity.deal.url}"
        )
        self.push(text)

    def craft_message(
        self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        prompt = (
            "Summarize this deal in 2-3 sentences for a push notification.\n"
            f"Item: {description}\nPrice: {deal_price}\nEst. value: {estimated_true_value}\n"
            "Respond only with the message."
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
        )
        usage.TRACKER.record(self.MODEL, getattr(response, "usage", None))
        return response.choices[0].message.content

    def notify(
        self,
        description: str,
        deal_price: float,
        estimated_true_value: float,
        url: str,
    ):
        self.log("Crafting message with LLM")
        text = self.craft_message(description, deal_price, estimated_true_value)
        self.push(text[:200] + "... " + url)

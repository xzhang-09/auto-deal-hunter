import os
from agents.deals import Opportunity
from agents.agent import Agent
from litellm import completion
import requests

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = "gpt-4o-mini"

    def __init__(self):
        self.log("Initializing")
        self.pushover_user = os.getenv("PUSHOVER_USER", "")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "")
        self.log("Ready")

    def push(self, text: str):
        self.log("Sending push notification")
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": text,
            "sound": "cashregister",
        }
        if self.pushover_user and self.pushover_token:
            requests.post(PUSHOVER_URL, data=payload)
        else:
            self.log("Pushover not configured - logging instead")
            self.log(text[:200])

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
        response = completion(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
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

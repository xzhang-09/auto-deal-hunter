import json
from typing import Optional, List
from openai import OpenAI
from agents.deals import ScrapedDeal, DealSelection
from agents.agent import Agent


class ScannerAgent(Agent):
    MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list.
Respond strictly in JSON with no explanation. Provide price as a number. If price isn't clear, exclude that deal.
Select deals with the most detailed product description and clear price."""

    USER_PROMPT_PREFIX = """Select the 5 most promising deals with detailed descriptions and clear price > 0.
Rephrase descriptions to summarize the product, not deal terms.
Deals:
"""

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        self.log("Initializing")
        self.openai = OpenAI()
        self.log("Ready")

    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        self.log("Fetching deals from RSS")
        urls = [opp.deal.url for opp in memory]
        scraped = ScrapedDeal.fetch()
        result = [s for s in scraped if s.url not in urls]
        self.log(f"Received {len(result)} new deals")
        return result

    def make_user_prompt(self, scraped) -> str:
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([s.describe() for s in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List = []) -> Optional[DealSelection]:
        scraped = self.fetch_deals(memory)
        if not scraped:
            return None
        user_prompt = self.make_user_prompt(scraped)
        self.log("Calling OpenAI with Structured Outputs")
        try:
            result = self.openai.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=DealSelection,
            )
            parsed = result.choices[0].message.parsed
        except AttributeError:
            result = self.openai.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT + " Respond with valid JSON only."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(result.choices[0].message.content or "{}")
            parsed = DealSelection(**data)
        parsed.deals = [d for d in parsed.deals if d.price > 0]
        self.log(f"Selected {len(parsed.deals)} deals with price>0")
        return parsed

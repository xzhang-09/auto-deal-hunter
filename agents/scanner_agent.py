import json
from typing import Optional, List
from openai import OpenAI
from agents.agent import Agent
from app.config import LLM_MODEL, LLM_SEED, LLM_TEMPERATURE
from app import usage
from models.deals import ScrapedDeal, DealSelection, deal_id


class ScannerAgent(Agent):
    MODEL = LLM_MODEL

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed new retail deals from a list.
Respond strictly in JSON with no explanation. Provide price as a number. Include list_price when supplied; use null when unknown. If price isn't clear, exclude that deal.
Exclude used, refurbished, renewed, open-box, pre-owned, scratch-and-dent, or otherwise non-new items.
Select deals with the most detailed product description and clear price."""

    USER_PROMPT_PREFIX = """Select the 5 most promising new retail deals with detailed descriptions and clear price > 0.
Exclude used, refurbished, renewed, open-box, pre-owned, scratch-and-dent, or otherwise non-new items.
Rephrase descriptions to summarize the product, not deal terms.
Preserve the provided List Price as list_price when it is a dollar value; otherwise use null.
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
        seen_ids = {deal_id(opp.deal.url) for opp in memory}
        scraped = ScrapedDeal.fetch()
        result = [s for s in scraped if deal_id(s.url) not in seen_ids]
        self.log(f"Received {len(result)} new deals")
        return result

    def make_user_prompt(self, scraped) -> str:
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([s.describe() for s in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: Optional[List] = None) -> Optional[DealSelection]:
        memory = memory or []
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
                temperature=LLM_TEMPERATURE,
                seed=LLM_SEED,
            )
            usage.TRACKER.record(self.MODEL, getattr(result, "usage", None))
            parsed = result.choices[0].message.parsed
        except AttributeError:
            result = self.openai.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT + " Respond with valid JSON only."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=LLM_TEMPERATURE,
                seed=LLM_SEED,
            )
            usage.TRACKER.record(self.MODEL, getattr(result, "usage", None))
            data = json.loads(result.choices[0].message.content or "{}")
            parsed = DealSelection(**data)
        parsed.deals = [d for d in parsed.deals if d.price > 0]
        # list_price comes from the scraper's structured extraction (price widget / meta), not
        # the LLM. Re-attach it by product id so gpt-4o-mini can't drop or alter it; deal_id is
        # robust to the slug changing between the scan and the model's echoed URL.
        scraped_by_id = {deal_id(s.url): s for s in scraped}
        for deal in parsed.deals:
            source = scraped_by_id.get(deal_id(deal.url))
            if source is not None:
                deal.list_price = source.list_price
        self.log(f"Selected {len(parsed.deals)} deals with price>0")
        return parsed

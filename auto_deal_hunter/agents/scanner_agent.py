from typing import Optional, List
from openai import OpenAI
from auto_deal_hunter.agents.agent import Agent
from auto_deal_hunter.infra.config import LLM_MAX_RETRIES, SCANNER_MODEL
from auto_deal_hunter.infra.openai_compat import parse_structured
from auto_deal_hunter.core.identity_policy import per_unit_fields
from auto_deal_hunter.core.source_ids import deal_id
from auto_deal_hunter.domain.deal import DealSelection
from auto_deal_hunter.ingest.scraper import ScrapedDeal


class ScannerAgent(Agent):
    MODEL = SCANNER_MODEL

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
        self.openai = OpenAI(max_retries=LLM_MAX_RETRIES)
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
        parsed = parse_structured(
            self.openai,
            model=self.MODEL,
            instructions=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            text_format=DealSelection,
        )
        parsed.deals = [d for d in parsed.deals if d.price > 0]
        # list_price comes from the scraper's structured extraction (price widget / meta), not
        # the LLM. Re-attach it by product id so gpt-4o-mini can't drop or alter it; deal_id is
        # robust to the slug changing between the scan and the model's echoed URL.
        scraped_by_id = {deal_id(s.url): s for s in scraped}
        for deal in parsed.deals:
            source = scraped_by_id.get(deal_id(deal.url))
            if source is not None:
                deal.list_price = source.list_price
                # Rebase a multipack to per-unit so it is valued against per-unit comparables.
                # Uses the scraped identity (not the model) so the pack size can't be altered.
                identity = getattr(source, "identity", None)
                (
                    deal.price,
                    deal.list_price,
                    deal.product_description,
                    deal.quantity,
                ) = per_unit_fields(deal.price, deal.list_price, deal.product_description, identity)
        self.log(f"Selected {len(parsed.deals)} deals with price>0")
        return parsed

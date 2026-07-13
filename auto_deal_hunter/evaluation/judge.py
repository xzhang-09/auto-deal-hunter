import re

from openai import OpenAI
from pydantic import BaseModel, Field

from auto_deal_hunter.domain.deal import Opportunity
from auto_deal_hunter.infra.config import JUDGE_MODEL, LLM_MAX_RETRIES
from auto_deal_hunter.infra.openai_compat import parse_structured


class MessageVerdict(BaseModel):
    faithful: bool = Field(description="Whether the message is faithful to the supplied deal data")
    issues: list[str] = Field(default_factory=list, description="Specific factual issues found")
    score: int = Field(description="Faithfulness score from 1 (bad) to 5 (excellent)")


_PRICE_RE = re.compile(r"\$(\d[\d,]*(?:\.\d{1,2})?)")


def _scale_first_price(message: str, factor: float) -> str | None:
    """Rescale the first dollar amount in the message, or None when it has no dollar amount."""
    match = _PRICE_RE.search(message)
    if match is None:
        return None
    value = float(match.group(1).replace(",", "")) * factor
    return f"{message[: match.start()]}${value:,.2f}{message[match.end():]}"


def corrupted_variants(message: str) -> dict[str, str]:
    """Deliberately unfaithful variants of a generated message.

    A judge that has only ever seen (and passed) clean messages is unvalidated: 100%%
    faithful could mean good messages or a judge that never flags anything. These
    deterministic corruptions provide the negative control — each misstates the deal in a
    way the judge is supposed to catch, so the catch rate measures judge recall while the
    clean pass rate bounds its false-positive rate."""
    variants: dict[str, str] = {}
    halved = _scale_first_price(message, 0.5)
    if halved is not None:
        variants["halved_price"] = halved  # understates price -> overstates savings
    doubled = _scale_first_price(message, 2.0)
    if doubled is not None:
        variants["doubled_price"] = doubled  # overstates the price/value it names
    variants["invented_fact"] = (
        message.rstrip() + " Includes a free 5-year extended warranty and a $50 gift card."
    )
    return variants


class ScanVerdict(BaseModel):
    faithful: bool = Field(description="Whether the scanner's summary and price match the raw listing")
    issues: list[str] = Field(default_factory=list, description="Specific factual issues found")
    score: int = Field(description="Faithfulness score from 1 (bad) to 5 (excellent)")


class ScanJudge:
    """Judges whether a scanner-selected deal is faithful to the raw scraped listing.

    Used by the model-tiering experiment (scripts/compare_scanner_models.py): a cheaper
    scanner model is only acceptable if its summaries and extracted prices stay grounded
    in the listing, and that has no numeric ground truth -- hence a judge."""

    def __init__(self, client=None, model: str = JUDGE_MODEL):
        self.client = client or OpenAI(max_retries=LLM_MAX_RETRIES)
        self.model = model

    def judge(self, listing_text: str, deal) -> ScanVerdict:
        return parse_structured(
            self.client,
            model=self.model,
            user_prompt=self._prompt(listing_text, deal),
            text_format=ScanVerdict,
        )

    @staticmethod
    def _prompt(listing_text: str, deal) -> str:
        quantity = getattr(deal, "quantity", None) or 1
        quantity_note = (
            f"- Pack size: {quantity} (the price above is PER UNIT, rebased from the pack price)\n"
            if quantity > 1
            else ""
        )
        return (
            "You are judging whether a deal scanner's output is faithful to the raw listing "
            "it was extracted from.\n"
            "Check that the summary describes the same product without inventing specs, and "
            "that the extracted price appears in (or follows arithmetically from) the listing. "
            "Summarization and omission are fine; contradiction and fabrication are not.\n\n"
            "Raw listing:\n"
            f"{listing_text}\n\n"
            "Scanner output:\n"
            f"- Summary: {deal.product_description}\n"
            f"- Price: ${deal.price:.2f}\n"
            f"{quantity_note}"
        )


class MessageJudge:
    def __init__(self, client=None, model: str = JUDGE_MODEL):
        self.client = client or OpenAI(max_retries=LLM_MAX_RETRIES)
        self.model = model

    def judge(self, opportunity: Opportunity, message: str) -> MessageVerdict:
        prompt = self._prompt(opportunity, message)
        return parse_structured(
            self.client,
            model=self.model,
            user_prompt=prompt,
            text_format=MessageVerdict,
        )

    @staticmethod
    def _prompt(opportunity: Opportunity, message: str) -> str:
        deal = opportunity.deal
        return (
            "You are judging whether a deal notification is faithful to the supplied data.\n"
            "Check that the price, estimate, savings, and product facts are not misstated. "
            "The message may be concise, but it must not invent unavailable details or overstate savings.\n\n"
            "Deal data:\n"
            f"- Product: {deal.product_description}\n"
            f"- Deal price: ${deal.price:.2f}\n"
            f"- Estimated fair value: ${opportunity.estimate:.2f}\n"
            f"- List price: {f'${deal.list_price:.2f}' if deal.list_price is not None else 'unknown'}\n"
            f"- Capped savings: ${opportunity.discount:.2f}\n"
            f"- URL: {deal.url}\n\n"
            f"Message:\n{message}\n"
        )

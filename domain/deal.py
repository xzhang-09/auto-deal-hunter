"""Core domain models for deals and opportunities (pure Pydantic types).

These are pure data structures with no I/O and no site-specific knowledge. Scraping and price
extraction live in the ``ingest`` package; the stable product-id helper (which knows about
each marketplace's URL format) lives in ``core.source_ids``. This module is safe to import
anywhere without pulling in network deps.
"""
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field


class Deal(BaseModel):
    product_description: str = Field(description="Summary of the product in 3-4 sentences")
    price: float = Field(description="Actual price of the product; per-unit when quantity > 1")
    list_price: Optional[float] = Field(default=None, description="DealNews original/list price when available; per-unit when quantity > 1")
    url: str = Field(description="URL of the deal")
    quantity: int = Field(default=1, description="Pack size for a multipack; price/list_price are per-unit when > 1")


class DealSelection(BaseModel):
    deals: List[Deal] = Field(description="5 deals with detailed descriptions and clear prices")


class Opportunity(BaseModel):
    deal: Deal
    estimate: float
    retrieval_confidence: Optional[float] = Field(
        default=None,
        description="Nearest-comparable RAG confidence in [0, 1], when recorded by the pricer",
    )

    @property
    def effective_value(self) -> float:
        """The estimate, capped at the seller's list price. A new-retail item's fair value
        cannot exceed its MSRP, so the estimate may only pull the value *below* list price,
        never above it — an over-confident estimate can't manufacture a discount above MSRP.
        With no list price to check against, the raw estimate is used."""
        if self.deal.list_price is not None:
            return min(self.estimate, self.deal.list_price)
        return self.estimate

    @computed_field
    @property
    def discount(self) -> float:
        """Defensible savings: the list-price-capped value minus the deal price. Kept as a
        computed field so it is still serialized, but it can never exceed list_price - price.
        Per-unit when the deal is a multipack (quantity > 1); see ``total_discount``."""
        return self.effective_value - self.deal.price

    @property
    def total_discount(self) -> float:
        """Capped savings scaled by pack size: total dollars saved across all units. Equals
        ``discount`` for a single item (quantity 1); for a multipack it scales the per-unit
        discount so whole-pack savings rank fairly against single items in ``core.scoring``."""
        return self.discount * self.deal.quantity

    @property
    def is_overestimate(self) -> bool:
        """True when the independent estimate exceeds the seller's list price.

        The pricer never sees list_price, so the estimate stays independent; list_price is
        only used here as a downstream sanity bound. A new-retail item's fair value should
        not exceed its original/MSRP price, so an estimate above it flags a likely upward
        bias in the pricer for dashboard reporting. Unknown list_price (None) is treated
        as "cannot check" rather than a failure."""
        return self.deal.list_price is not None and self.estimate > self.deal.list_price

    def is_comparables_mismatch(self, ratio: float) -> bool:
        """True when the estimate exceeds ``ratio`` times the list price -- the signature of a
        retrieval mismatch rather than ordinary estimator noise.

        A slightly-high estimate (a few percent over list) is normal upward error; an estimate
        at a multiple of list price means the RAG neighbors were the wrong kind of product
        (e.g. battery chargers retrieved for alkaline batteries), so no trustworthy estimate
        exists. Callers use this to zero the push confidence and blank the displayed estimate;
        the stored estimate and ``is_overestimate`` monitoring stay untouched. The threshold
        comes from config (``ESTIMATE_MISMATCH_RATIO``); this stays a pure predicate."""
        return self.deal.list_price is not None and self.estimate > ratio * self.deal.list_price

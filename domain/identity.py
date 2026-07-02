"""Product-identity model for deal disambiguation.

Pure data, no I/O. The RAG pricer values a product by comparing it to similar items, so a
listing that is really a multi-pack, a heterogeneous bundle, or a subscription would be
priced against single-unit comparables and manufacture a bogus discount. ``ProductIdentity``
captures *what is actually being sold* so a downstream policy (``core.identity_policy``) can
decide whether the item is safely priceable or should be skipped. The deterministic extractor
lives in ``ingest.identity``; this module only defines the shape.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ItemKind(str, Enum):
    SINGLE = "single"              # one unit of one product -> priceable as-is
    MULTIPACK = "multipack"        # N identical units (e.g. "36-Pack") -> per-unit only
    BUNDLE = "bundle"              # heterogeneous items sold together -> not a single value
    SUBSCRIPTION = "subscription"  # recurring/term pricing -> not a one-off retail price
    AGGREGATOR = "aggregator"      # sale/roundup/coupon page, not a single product at all
    UNKNOWN = "unknown"            # could not be determined -> treat conservatively


class ProductIdentity(BaseModel):
    kind: ItemKind = ItemKind.SINGLE
    quantity: int = Field(default=1, description="Units in a multipack; 1 otherwise")
    variant: Optional[str] = Field(
        default=None, description="Price-affecting spec found in the text, e.g. '512GB', '55-inch'"
    )
    bundle_components: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="rule", description="'rule' | 'llm'")

"""Deterministic selection of the single best opportunity.

The LLM agent's job is to *gather* candidates (scan deals, estimate each value). Deciding
which deal is "most compelling" is a pure ranking and should NOT be left to model judgment:
a deterministic ``max`` over the list-price-capped discount is reproducible and provably
optimal, where the model might pick a worse deal. ``Opportunity.discount`` already applies
the MSRP cap, so ranking by it reuses the same guardrail the dashboard reports.
"""
from __future__ import annotations

from typing import Iterable, Optional

from models.deals import Opportunity


def rank_opportunities(candidates: Iterable[Opportunity]) -> list[Opportunity]:
    """Sort candidates by capped discount, highest first. Stable for equal discounts."""
    return sorted(candidates, key=lambda o: o.discount, reverse=True)


def best_opportunity(
    candidates: Iterable[Opportunity], min_discount: float = 0.0
) -> Optional[Opportunity]:
    """The highest-discount candidate whose capped discount clears ``min_discount``.

    Returns None when no candidate is a genuine bargain (discount <= min_discount), which
    is the correct "notify nobody" outcome rather than surfacing a non-deal.
    """
    ranked = rank_opportunities(candidates)
    if ranked and ranked[0].discount > min_discount:
        return ranked[0]
    return None

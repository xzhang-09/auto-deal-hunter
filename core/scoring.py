"""Deterministic selection of the single best opportunity.

The LLM agent's job is to *gather* candidates (scan deals, estimate each value). Deciding
which deal is "most compelling" is a pure ranking and should NOT be left to model judgment:
a deterministic ``max`` over the list-price-capped savings is reproducible and provably
optimal, where the model might pick a worse deal. ``Opportunity.total_discount`` applies the
MSRP cap (reusing the guardrail the dashboard reports) and scales by pack size, so a
multipack's whole-pack savings rank fairly against single items.
"""
from __future__ import annotations

from typing import Iterable, Optional

from domain.deal import Opportunity


def rank_opportunities(candidates: Iterable[Opportunity]) -> list[Opportunity]:
    """Sort candidates by total capped savings, highest first. Stable for equal savings."""
    return sorted(candidates, key=lambda o: o.total_discount, reverse=True)


def best_opportunity(
    candidates: Iterable[Opportunity], min_discount: float = 0.0
) -> Optional[Opportunity]:
    """The highest-savings candidate whose total capped savings clears ``min_discount``.

    Returns None when no candidate is a genuine bargain (savings <= min_discount), which
    is the correct "notify nobody" outcome rather than surfacing a non-deal.
    """
    ranked = rank_opportunities(candidates)
    if ranked and ranked[0].total_discount > min_discount:
        return ranked[0]
    return None

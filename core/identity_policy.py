"""Deterministic policy: given a product identity, decide if the item is safely priceable.

Selection and gating are kept out of the LLM's hands (same rationale as ``core.scoring``):
whether a multipack/bundle/subscription can be valued against single-unit comparables is a
fixed rule, not a judgment call. Skipping an ambiguous listing ("少而准 > 多而疑") is the
correct outcome -- a missed deal costs nothing, a surfaced bogus deal costs trust.

Multipacks are priceable: the comparable side records pack size (see build_vector_store) so
both the deal and its comparables can be rebased to a per-unit price -- see ``per_unit_fields``.
Bundles and subscriptions remain non-priceable, since they cannot be valued as a single unit.
"""
from __future__ import annotations

from typing import Optional, Tuple

from domain.identity import ItemKind, ProductIdentity

# Below this, a rule-extracted identity is too weak to act on; treat as a plain single item.
CONFIDENCE_FLOOR = 0.5

# Kinds that cannot be valued as a one-off single-unit retail price.
_NOT_PRICEABLE = {
    ItemKind.BUNDLE,
    ItemKind.SUBSCRIPTION,
    ItemKind.AGGREGATOR,
    ItemKind.UNKNOWN,
}


def per_unit_note(quantity: int) -> str:
    """Suffix appended to a multipack's description marking its price as per single unit.
    Shared so the UI can strip it again when it shows pack-level prices to the user."""
    return f" (per-unit price; sold in packs of {quantity})"


def per_unit_fields(
    price: float,
    list_price: Optional[float],
    description: str,
    identity: Optional[ProductIdentity],
) -> Tuple[float, Optional[float], str, int]:
    """Rebase a multipack listing onto a single-unit basis.

    Returns ``(price, list_price, description, quantity)``. A pack of N is divided to a
    per-unit price (and list price) so it is valued against the pricer's per-unit comparables,
    and the description is annotated so both the estimate and the user-facing numbers read as
    per-unit. Non-multipacks pass through unchanged with quantity 1.
    """
    if identity is None or identity.kind is not ItemKind.MULTIPACK or identity.quantity <= 1:
        return price, list_price, description, 1
    quantity = identity.quantity
    per_unit_list = list_price / quantity if list_price is not None else None
    return (
        price / quantity,
        per_unit_list,
        description + per_unit_note(quantity),
        quantity,
    )


def resolve(identity: Optional[ProductIdentity]) -> Tuple[str, str]:
    """Return ``(action, reason)`` where action is ``"price"`` or ``"abstain"``.

    ``None`` (no identity signal) and low-confidence identities are treated as plain single
    items -- the safe default that preserves current behavior for ordinary listings.
    """
    if identity is None or identity.confidence < CONFIDENCE_FLOOR:
        return "price", "no confident identity signal; treated as single item"
    if identity.kind in _NOT_PRICEABLE:
        return "abstain", f"not priceable as a single item (kind={identity.kind.value})"
    return "price", "single item"


def is_priceable(identity: Optional[ProductIdentity]) -> bool:
    action, _ = resolve(identity)
    return action == "price"

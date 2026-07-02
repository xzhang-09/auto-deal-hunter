"""Deterministic product-identity extraction from listing text.

Mirrors the cheap, testable heuristic style of ``is_non_new_condition`` next door: a regex
pass that catches the common, unambiguous cases (multi-packs, subscriptions, obvious bundles,
storage/screen variants) at zero cost and with full reproducibility. Anything it cannot
classify returns ``None`` so the caller treats it as a plain single item (the safe default) --
or, in a later phase, hands it to an LLM extractor for the ambiguous long tail.
"""
import re
from typing import Optional

from domain.identity import ItemKind, ProductIdentity

# "36-Pack", "2 Pack", "10-Count", "24 ct", "4pk", "5 Pieces", "Pack of 3", "Set of 2".
# Requires an explicit pack word so model names like "4K"/"65W" do not register. The number is
# guarded against being a model/part-number fragment, which previously produced absurd pack
# sizes (e.g. "HP 8121-0840 ... PK Power" -> 840, "CT-90325 Remote" -> 90325):
#   (?<![\d.-])  the digits must not be the tail of a longer number/part code (8121-0840)
#   (?!0)        no leading zero -- real counts are not written "0840"
#   \d{1,4}      at most 4 digits, so a long part number can't be read as a quantity and the
#                pack size is structurally bounded (legit bulk like a 1000/5000-pack survives)
#   [ -]?        a single space or hyphen only -- never a newline, so a part code at a line end
#                cannot bind to a brand ("...pk") on the next line
_MULTIPACK = re.compile(
    r"(?<![\d.-])(?!0)(\d{1,4})[ -]?(?:pack|count|pcs|pieces|pk|ct)\b"
    r"|\bpack of (\d+)\b|\bset of (\d+)\b",
    re.I,
)
# Recurring/term pricing. Kept conservative: "per month"/"/mo"/"/month"/"subscription"/
# "membership", not bare "monthly" (which appears in benign marketing) or "12-month warranty".
_SUBSCRIPTION = re.compile(
    r"\bsubscription\b|\bmembership\b|\bper month\b|/mo(?:nth)?\b|\bannual plan\b|\b\d+[\s-]?month plan\b",
    re.I,
)
# Heterogeneous bundle signal. High precision on purpose: only the explicit word "bundle".
# An audit of live listings showed looser signals ("combo", "with free", "+ free") fire on
# product names ("Combo Robot Vacuum") and the ubiquitous "+ free shipping", which flagged
# ~91% of deals as bundles. A real bundle we miss falls back to single-item valuation; a
# false bundle wrongly skips a genuine deal, so erring toward precision is the safer default.
_BUNDLE = re.compile(r"\bbundles?\b", re.I)
# Sale/roundup/coupon landing pages -- not a single product, so they have no one price to
# value. Anchored on phrases a single listing never carries: a discount *range* ("up to N%
# off"), a store-wide/coupon offer, or a plural "Deals:" / "Sale:" roundup heading. An audit
# of live listings showed these are common and would otherwise be priced as if they were one
# item, producing a junk estimate that can win the deterministic best-deal selection.
_AGGREGATOR = re.compile(
    r"\bup to\b[^.]*?\d+%\s*off\b"      # "up to 45% off", "up to an extra 70% off"
    r"|\b\d+%\s*off\s+sitewide\b"
    r"|\bpromo code\b"
    r"|\bclearance hub\b"
    r"|\btop \d+ deals\b"
    r"|\bdeals\s*:"
    r"|\bdeals\s+(?:for\s+)?from\b"
    r"|\bsale\s*:",
    re.I,
)
# Price-affecting variants worth recording (used for retrieval/consistency in a later phase).
_STORAGE = re.compile(r"\b(\d+)\s?(TB|GB)\b", re.I)
_SCREEN = re.compile(r"\b(\d{2,3})[\s-]?(?:inch|\")", re.I)


def _multipack_quantity(text: str) -> Optional[int]:
    match = _MULTIPACK.search(text)
    if not match:
        return None
    qty = int(next(g for g in match.groups() if g))
    return qty if qty >= 2 else None


def _variant(text: str) -> Optional[str]:
    if (m := _STORAGE.search(text)):
        return m.group(0).replace(" ", "").upper()
    if (m := _SCREEN.search(text)):
        return m.group(0)
    return None


def extract_identity_rule(text: str) -> Optional[ProductIdentity]:
    """Best-effort deterministic identity. ``None`` means "no signal" -> treat as single.

    Precedence (most disqualifying first): aggregator (not a product at all), subscription,
    multipack, bundle. A recognized price-affecting variant on an otherwise plain item is
    recorded as a SINGLE so it can be used downstream without changing priceability.
    """
    if _AGGREGATOR.search(text):
        return ProductIdentity(kind=ItemKind.AGGREGATOR, confidence=0.9)
    if _SUBSCRIPTION.search(text):
        return ProductIdentity(kind=ItemKind.SUBSCRIPTION, confidence=0.9)

    qty = _multipack_quantity(text)
    if qty is not None:
        return ProductIdentity(kind=ItemKind.MULTIPACK, quantity=qty, confidence=0.85)

    variant = _variant(text)
    if _BUNDLE.search(text):
        return ProductIdentity(kind=ItemKind.BUNDLE, variant=variant, confidence=0.6)
    if variant is not None:
        return ProductIdentity(kind=ItemKind.SINGLE, variant=variant, confidence=0.8)
    return None

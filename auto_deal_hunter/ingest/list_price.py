"""Price extraction from DealNews listings.

Pure text/HTML parsing: given a deal's text (or its parsed price widget), recover the deal
price and the original/list price. No network I/O — this is the regex/BeautifulSoup layer
that the scraper feeds. Kept separate from ``scraper`` so the extraction heuristics can be
unit-tested in isolation (see tests/test_extract_list_price.py)."""
import re
from typing import Optional

PRICE_PATTERN = r"\$?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
LIST_PRICE_PATTERNS = [
    rf"\b(?:list|regular|original|retail|was)\s+price\s*:?\s*{PRICE_PATTERN}",
    rf"\b(?:list|regular|original|retail|was)\s+price\s+of\s+{PRICE_PATTERN}",
    rf"\b(?:list|regular|original|retail|was)\s*:?\s*{PRICE_PATTERN}",
    rf"\bMSRP\s*:?\s*{PRICE_PATTERN}",
    rf"\bfrom\s+(?:its\s+)?{PRICE_PATTERN}\s+(?:list|regular|original|retail)\s+price\b",
    rf"\bfrom\s+(?:a\s+)?(?:regular|original|retail)\s+price\s+of\s+{PRICE_PATTERN}",
    rf"\(\s*from\s+{PRICE_PATTERN}\s*\)",
]
# DealNews frequently states the saving relative to list price ("It's $150 under list price")
# instead of printing the list price itself, often only in the page's meta description.
UNDER_LIST_PRICE_PATTERN = (
    rf"{PRICE_PATTERN}\s+(?:under|off|below)\s+(?:the\s+)?(?:list|regular|original|retail|msrp)\b"
)
DOLLAR_AMOUNT_PATTERN = r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"


def parse_money(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _money_regex(value: float) -> str:
    dollars = int(value)
    cents = int(round((value - dollars) * 100))
    base = f"{dollars:,}|{dollars}"
    if cents:
        whole_values = {dollars, int(round(value))}
        whole_patterns = [rf"(?:{whole:,}|{whole})(?:\.00)?" for whole in whole_values]
        return rf"\$\s*(?:(?:{base})\.{cents:02d}|{'|'.join(whole_patterns)})(?![0-9]|,[0-9])"
    return rf"\$\s*(?:{base})(?:\.00)?(?![0-9]|,[0-9])"


def _higher_than_deal(candidate: Optional[float], deal_price: Optional[float]) -> bool:
    return candidate is not None and (deal_price is None or candidate > deal_price)


def _stacked_discount_list_price(text: str, deal_price: Optional[float]) -> Optional[float]:
    if not deal_price:
        return None
    text = re.split(r"\bRelated Offers\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    percentages = [
        float(match)
        for match in re.findall(r"\b(?:extra\s+)?([0-9]{1,2}(?:\.[0-9]+)?)%\s+off\b", text, flags=re.IGNORECASE)
    ]
    if len(percentages) < 2:
        return None
    multiplier = 1.0
    for percent in percentages:
        multiplier *= 1 - (percent / 100)
    if multiplier <= 0:
        return None
    derived = deal_price / multiplier
    money_values = [parse_money(match) for match in re.findall(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text)]
    money_values = [value for value in money_values if value and value > deal_price]
    for value in money_values:
        if abs(value - derived) / derived <= 0.03:
            return value
    return round(derived, 2)


def _under_list_price(text: str, deal_price: Optional[float]) -> Optional[float]:
    """Reconstruct the list price from a savings callout like "$150 under list price":
    list_price = deal_price + the stated delta. Needs the deal price as the baseline."""
    if not deal_price:
        return None
    match = re.search(UNDER_LIST_PRICE_PATTERN, text, flags=re.IGNORECASE)
    if not match:
        return None
    delta = parse_money(match.group(1))
    if delta and delta > 0:
        return round(deal_price + delta, 2)
    return None


def extract_list_price(text: str, deal_price: Optional[float] = None) -> Optional[float]:
    primary_text = re.split(r"\bRelated Offers\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    normalized = " ".join(primary_text.split())
    for pattern in LIST_PRICE_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            candidate = parse_money(match.group(1))
            if _higher_than_deal(candidate, deal_price):
                return candidate
    candidate = _under_list_price(normalized, deal_price)
    if candidate:
        return candidate
    if deal_price:
        price_pattern = _money_regex(deal_price)
        promo_terms = r"(?:\s+(?!\$)[A-Za-z0-9&/+.'-]+){0,8}"
        double_price_pattern = rf"{price_pattern}{promo_terms}\s+\$([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)"
        for match in re.finditer(double_price_pattern, normalized, flags=re.IGNORECASE):
            candidate = parse_money(match.group(1))
            if _higher_than_deal(candidate, deal_price):
                return candidate
    candidate = _stacked_discount_list_price(normalized, deal_price)
    if candidate:
        return candidate
    return None


def extract_deal_price(text: str) -> Optional[float]:
    match = re.search(PRICE_PATTERN, text)
    return parse_money(match.group(1)) if match else None


def extract_callout_prices(soup) -> tuple[Optional[float], Optional[float]]:
    """Read DealNews's structured price widget. The headline ``.callout-group`` holds the
    deal price (``.callout``) and the struck-through original/list price (``.callout-comparison``).
    Returns ``(deal_price, list_price)``, either of which may be None.

    This is the most reliable source: it is independent of body wording and immune to model
    numbers in the title being mistaken for a price (e.g. "Archer BE9300")."""
    group = soup.find(class_="callout-group")
    if not group:
        return None, None
    deal = list_price = None
    callout = group.find(class_="callout")
    if callout:
        match = re.search(DOLLAR_AMOUNT_PATTERN, callout.get_text(" ", strip=True))
        deal = parse_money(match.group(1)) if match else None
    comparison = group.find(class_="callout-comparison")
    if comparison:
        match = re.search(DOLLAR_AMOUNT_PATTERN, comparison.get_text(" ", strip=True))
        list_price = parse_money(match.group(1)) if match else None
    return deal, list_price

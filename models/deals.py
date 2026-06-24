from pydantic import BaseModel, Field, computed_field
from typing import List, Dict, Optional
from typing_extensions import Self
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time
import logging

from app import http_cache

FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

NON_NEW_PATTERNS = [
    r"\brefurb(?:ished)?\b",
    r"\brenewed\b",
    r"\bopen[\s-]?box\b",
    r"\bpre[\s-]?owned\b",
    r"\bscratch and dent\b",
    r"\bused\s+(?:item|condition|product|device|laptop|phone|tablet|watch|speaker|camera|pc|monitor|headphones?)\b",
    r"\bcondition\s*:\s*used\b",
]
USED_CONDITION_PATTERN = r"\bused\b"
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


def deal_id(url: str) -> str:
    """DealNews's numeric product id from a deal URL, robust to slug text changing
    between RSS pulls (e.g. a typo fix from "200-W" to "2000-W") while the id stays put."""
    match = re.search(r"/(\d+)\.html", url)
    return match.group(1) if match else url


def is_non_new_condition(text: str) -> bool:
    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in NON_NEW_PATTERNS)


def extract(html_snippet: str) -> str:
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")
    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace("\n", " ")


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


DOLLAR_AMOUNT_PATTERN = r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"


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


class ScrapedDeal:
    def __init__(self, entry: Dict[str, str]):
        self.title = entry["title"]
        self.summary = extract(entry["summary"])
        self.price = extract_deal_price(" ".join([self.title, self.summary]))
        self.url = entry["links"][0]["href"]
        # Read-through page cache: avoid re-fetching the same DealNews page on every run
        # (RSS feeds overlap and runs repeat). On a miss, fetch live and store the bytes.
        stuff = http_cache.read(self.url)
        if stuff is None:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            stuff = response.content
            http_cache.write(self.url, stuff)
        soup = BeautifulSoup(stuff, "html.parser")
        content = soup.find("div", class_="content-section")
        content = content.get_text() if content else ""
        content = content.replace("\nmore", "").replace("\n", " ")
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""
        # DealNews puts its "$N under list price" savings callout in the meta description,
        # not the body, so read it too or relative list prices are missed at the source.
        meta_tag = soup.find("meta", attrs={"name": "description"})
        self.meta_description = meta_tag.get("content", "") if meta_tag else ""
        # The structured price widget is the most reliable source for both prices; prefer it.
        # Its deal price also corrects RSS titles where a model number ("BE9300") is mis-read
        # as the price. Fall back to text/meta parsing when the widget is absent.
        widget_price, widget_list_price = extract_callout_prices(soup)
        if widget_price:
            self.price = widget_price
        if widget_list_price and _higher_than_deal(widget_list_price, self.price):
            self.list_price = widget_list_price
        else:
            self.list_price = extract_list_price(
                " ".join([self.title, self.summary, self.details, self.features, self.meta_description]),
                deal_price=self.price,
            )
        self.truncate()

    def is_new_retail(self) -> bool:
        title = self.title.lower()
        condition_text = " ".join([self.title, self.summary, self.details, self.features])
        if is_non_new_condition(condition_text):
            return False
        return re.search(USED_CONDITION_PATTERN, title) is None

    def truncate(self):
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def describe(self):
        list_price = f"${self.list_price:.2f}" if self.list_price else "Unknown"
        return (
            f"Title: {self.title}\n"
            f"List Price: {list_price}\n"
            f"Details: {self.details.strip()}\n"
            f"Features: {self.features.strip()}\n"
            f"URL: {self.url}"
        )

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        deals = []
        feed_iter = tqdm(FEEDS) if show_progress else FEEDS
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                try:
                    deal = cls(entry)
                    if deal.is_new_retail():
                        deals.append(deal)
                    else:
                        logging.info("Skipping non-new deal: %s", deal.title)
                except Exception as exc:
                    logging.warning("Skipping deal after fetch/parse failure: %s", exc)
                time.sleep(0.05)
        return deals


class Deal(BaseModel):
    product_description: str = Field(description="Summary of the product in 3-4 sentences")
    price: float = Field(description="Actual price of the product")
    list_price: Optional[float] = Field(default=None, description="DealNews original/list price when available")
    url: str = Field(description="URL of the deal")


class DealSelection(BaseModel):
    deals: List[Deal] = Field(description="5 deals with detailed descriptions and clear prices")


class Opportunity(BaseModel):
    deal: Deal
    estimate: float

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
        computed field so it is still serialized, but it can never exceed list_price - price."""
        return self.effective_value - self.deal.price

    @property
    def is_overestimate(self) -> bool:
        """True when the independent estimate exceeds the seller's list price.

        The pricer never sees list_price, so the estimate stays independent; list_price is
        only used here as a downstream sanity bound. A new-retail item's fair value should
        not exceed its original/MSRP price, so an estimate above it flags a likely upward
        bias in the pricer for dashboard reporting. Unknown list_price (None) is treated
        as "cannot check" rather than a failure."""
        return self.deal.list_price is not None and self.estimate > self.deal.list_price

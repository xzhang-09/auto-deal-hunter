"""DealNews RSS scraping and new-retail filtering.

Fetches deal pages (through the on-disk page cache), parses the body/meta/price-widget, and
delegates price recovery to ``ingest.list_price``. This is the I/O layer; the pure parsing
heuristics live next door so they can be tested without the network."""
import logging
import re
import time
from typing import Dict, List

import feedparser
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing_extensions import Self

from core.identity_policy import is_priceable
from infra import http_cache
from ingest.identity import extract_identity_rule
from ingest.list_price import (
    _higher_than_deal,
    extract_callout_prices,
    extract_deal_price,
    extract_list_price,
)

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
        # Identity (single / multipack / bundle / subscription) decides whether this listing
        # can be valued against single-unit comparables. Derived from the TITLE only: the
        # marketing body is full of incidental mentions ("no subscription needed", "free
        # shipping") that an audit showed produce large false-positive rates, whereas the
        # title is DealNews's own concise statement of what is being sold.
        self.identity = extract_identity_rule(self.title)

    def is_new_retail(self) -> bool:
        title = self.title.lower()
        condition_text = " ".join([self.title, self.summary, self.details, self.features])
        if is_non_new_condition(condition_text):
            return False
        return re.search(USED_CONDITION_PATTERN, title) is None

    def is_priceable(self) -> bool:
        """False for multipacks/bundles/subscriptions, which the RAG pricer would value
        against single-unit comparables and so manufacture a bogus discount."""
        return is_priceable(self.identity)

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
                    if not deal.is_new_retail():
                        logging.info("Skipping non-new deal: %s", deal.title)
                    elif not deal.is_priceable():
                        kind = deal.identity.kind.value if deal.identity else "unknown"
                        logging.info("Skipping non-priceable deal (%s): %s", kind, deal.title)
                    else:
                        deals.append(deal)
                except Exception as exc:
                    logging.warning("Skipping deal after fetch/parse failure: %s", exc)
                time.sleep(0.05)
        return deals

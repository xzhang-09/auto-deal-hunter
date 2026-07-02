"""Audit the deterministic identity extractor against live DealNews listings.

Diagnostic only -- not part of the agent loop, no LLM or vector store needed. It answers one
question: *how big is the long tail?* i.e. the share of real listings that the regex extractor
(``ingest.identity``) classifies as ``none`` (no signal -> treated as a plain single item).

Why it matters: a ``none`` listing that is really a multipack / bundle / subscription is a
*missed* identity that will pollute the valuation. If a hand-scan of the dumped ``none``
samples turns up many such misses, an LLM extraction layer for the long tail is worth adding;
if ``none`` is almost all genuine single items, it is not. Run this before building that layer.

Usage:
  python scripts/audit_identity.py --per-feed 20 --output-json data/identity_audit.json

Then eyeball ``none_samples`` (or the JSON dump) and count the true misses by hand. Also skim
``classified_samples`` to check the regex is not over-firing (false positives).
"""
import argparse
import json
import logging
import time
from collections import Counter

import feedparser

from ingest.scraper import FEEDS, ScrapedDeal

# Identity kinds the regex can assign; "none" is the synthetic bucket for "no signal".
_CLASSIFIED_KINDS = ("multipack", "bundle", "subscription", "aggregator", "single")


def scrape_records(per_feed: int) -> list[dict]:
    """Scrape listings WITHOUT the priceability filter, recording each one's identity.

    Mirrors ``ScrapedDeal.fetch`` but keeps every deal (including ones the agent would skip)
    so the audit sees the full distribution, not just what survives filtering."""
    records = []
    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:per_feed]:
            try:
                deal = ScrapedDeal(entry)
            except Exception as exc:
                logging.warning("Skipping deal after fetch/parse failure: %s", exc)
                continue
            identity = deal.identity
            records.append(
                {
                    "title": deal.title,
                    "kind": identity.kind.value if identity else "none",
                    "quantity": identity.quantity if identity else 1,
                    "variant": identity.variant if identity else None,
                    "new_retail": deal.is_new_retail(),
                    "priceable": deal.is_priceable(),
                    "url": deal.url,
                }
            )
            time.sleep(0.05)
    return records


def summarize(records: list[dict], sample_limit: int = 25) -> dict:
    """Pure aggregation over scraped records (no I/O); see tests.

    Distribution is computed over *new-retail* listings only, since used/refurb items are
    filtered upstream and would otherwise dilute the long-tail estimate.
    """
    new_retail = [r for r in records if r["new_retail"]]
    n = len(new_retail)
    kinds = Counter(r["kind"] for r in new_retail)
    none_titles = [r["title"] for r in new_retail if r["kind"] == "none"]
    return {
        "scraped": len(records),
        "new_retail": n,
        "kind_counts": dict(kinds),
        "none_share": (kinds.get("none", 0) / n) if n else 0.0,
        "none_samples": none_titles[:sample_limit],
        "classified_samples": {
            kind: [r["title"] for r in new_retail if r["kind"] == kind][:sample_limit]
            for kind in _CLASSIFIED_KINDS
            if kind != "single"
        },
    }


def _print_report(stats: dict) -> None:
    n = stats["new_retail"]
    print(f"Scraped {stats['scraped']} listings; {n} new-retail.\n")
    print("Identity distribution (new-retail only):")
    for kind, count in sorted(stats["kind_counts"].items(), key=lambda kv: -kv[1]):
        share = count / n if n else 0.0
        print(f"  {kind:<13} {count:>4}  ({share:.0%})")
    print(f"\nLong-tail signal -> 'none' share: {stats['none_share']:.0%} of new-retail listings.")
    print("  (Hand-scan the 'none' samples below: count how many are really missed")
    print("   multipacks / bundles / subscriptions. Many misses => an LLM layer is worth it.)\n")
    if stats["none_samples"]:
        print("Sample 'none' (no signal) titles:")
        for title in stats["none_samples"]:
            print(f"  - {title}")
    for kind, titles in stats["classified_samples"].items():
        if titles:
            print(f"\nSample '{kind}' titles (check for false positives):")
            for title in titles:
                print(f"  - {title}")


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Audit identity extraction over live DealNews deals.")
    parser.add_argument("--per-feed", type=int, default=20, help="Listings per RSS feed (default: 20).")
    parser.add_argument("--output-json", help="Optional path to write full records + summary as JSON.")
    args = parser.parse_args()

    records = scrape_records(args.per_feed)
    stats = summarize(records)
    _print_report(stats)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"summary": stats, "records": records}, f, indent=2)
        print(f"\nWrote full audit to {args.output_json}")


if __name__ == "__main__":
    main()

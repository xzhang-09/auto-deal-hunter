"""Pure metric helpers for retriever-only evaluation.

Importable, unit-tested logic (see tests/test_eval_retrieval.py); the CLI that embeds the
holdout and queries the vector store lives in scripts/eval_retrieval.py.
"""
import statistics

# APE is undefined-in-practice when the true per-unit price approaches zero: with a $0.06
# denominator, a few dollars of neighbor-median movement swings the query's APE by thousands
# of percentage points, and a single such row once flipped the sign of a rerank A/B comparison
# (it alone accounted for the entire apparent meanAPE regression). Rows this cheap are junk
# per-unit prices (e.g. 200-packs of accessories), not products the pricer is expected to value,
# so they are excluded from the price-APE metric entirely; category metrics still count them.
APE_MIN_TRUTH_PRICE = 1.0


def _per_unit_price(record: dict):
    """Price on a single-unit basis, dividing a multipack by its recorded pack size.

    Keeps the price comparison consistent with the pricer's per-unit comparables: a 36-pack
    stored at its pack price would otherwise inflate the neighbor/true price. Records from
    stores built before quantity was recorded have no such key and default to quantity 1."""
    price = record.get("price")
    if price is None:
        return None
    return price / (record.get("quantity") or 1)


def retrieval_metrics(neighbors: list[dict], item: dict) -> dict[str, float]:
    """Per-query retrieval metrics given the retrieved neighbor metadatas and the query item."""
    same_category = [n for n in neighbors if n.get("category") == item["category"]]
    neighbor_prices = [p for n in neighbors if (p := _per_unit_price(n)) is not None]
    median_price = statistics.median(neighbor_prices) if neighbor_prices else None
    truth = _per_unit_price(item)
    ape_scorable = median_price is not None and truth is not None and truth >= APE_MIN_TRUTH_PRICE
    price_ape = abs(median_price - truth) / truth if ape_scorable else None
    return {
        "category_precision": len(same_category) / len(neighbors) if neighbors else 0.0,
        "hit": 1.0 if same_category else 0.0,
        "price_ape": price_ape,
    }


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    n = len(per_query)
    apes = [m["price_ape"] for m in per_query if m["price_ape"] is not None]
    return {
        "n": n,
        "n_ape": len(apes),
        "category_precision": sum(m["category_precision"] for m in per_query) / n,
        "hit_rate": sum(m["hit"] for m in per_query) / n,
        # Median is the headline: even with the APE_MIN_TRUTH_PRICE guard excluding near-zero
        # denominators, APE is ratio-scaled and long-tailed, so the mean chases outliers (a
        # single $0.01 row once drove it to 764% while the median stayed 32%). The mean is
        # kept alongside it to surface tail/outlier behavior; n_ape says how many queries
        # were APE-scorable after the guard.
        "price_median_ape": statistics.median(apes) if apes else float("nan"),
        "price_mape": (sum(apes) / len(apes)) if apes else float("nan"),
    }

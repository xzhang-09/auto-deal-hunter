import argparse
import os
from collections import defaultdict

from dotenv import load_dotenv

from core.opportunity_store import OpportunityStore
from infra.config import RAG_MIN_CONFIDENCE
from infra.paths import DEFAULT_DEALS_DB_PATH

load_dotenv(override=True)


def _precision(labels: list[str | None]) -> tuple[int, int, int, float]:
    good = sum(1 for label in labels if label == "good_deal")
    bad = sum(1 for label in labels if label == "bad_deal")
    labeled = good + bad
    return labeled, good, bad, (good / labeled if labeled else 0.0)


def confidence_bucket(confidence: float | None) -> str:
    # The first boundary is the live notify gate, so the report's lowest bucket always
    # answers "what did the RAG_MIN_CONFIDENCE gate hold back" even when the env overrides it.
    if confidence is None:
        return "unknown"
    if confidence < RAG_MIN_CONFIDENCE:
        return f"<{RAG_MIN_CONFIDENCE:.2f} (below notify gate)"
    if confidence < 0.4:
        return f"{RAG_MIN_CONFIDENCE:.2f}-0.39"
    if confidence < 0.7:
        return "0.40-0.69"
    return ">=0.70"


def discount_bucket(total_discount: float) -> str:
    if total_discount < 10:
        return "<$10"
    if total_discount < 50:
        return "$10-$49"
    if total_discount < 100:
        return "$50-$99"
    return ">=$100"


def summarize_feedback(rows) -> dict[str, object]:
    labels = [feedback for _, feedback in rows]
    labeled, good, bad, precision = _precision(labels)
    buckets: dict[str, dict[str, list[str | None]]] = {
        "confidence": defaultdict(list),
        "list_price": defaultdict(list),
        "overestimate": defaultdict(list),
        "discount": defaultdict(list),
    }
    for opportunity, feedback in rows:
        buckets["confidence"][confidence_bucket(opportunity.retrieval_confidence)].append(feedback)
        buckets["list_price"]["known" if opportunity.deal.list_price is not None else "unknown"].append(feedback)
        buckets["overestimate"]["yes" if opportunity.is_overestimate else "no"].append(feedback)
        buckets["discount"][discount_bucket(opportunity.total_discount)].append(feedback)
    return {
        "overall": {
            "total": len(rows),
            "labeled": labeled,
            "good": good,
            "bad": bad,
            "unlabeled": len(rows) - labeled,
            "precision": precision,
        },
        "buckets": {
            name: {
                bucket: {
                    "total": len(bucket_labels),
                    "labeled": _precision(bucket_labels)[0],
                    "good": _precision(bucket_labels)[1],
                    "bad": _precision(bucket_labels)[2],
                    "precision": _precision(bucket_labels)[3],
                }
                for bucket, bucket_labels in sorted(groups.items())
            }
            for name, groups in buckets.items()
        },
    }


def print_summary(summary: dict[str, object]) -> None:
    overall = summary["overall"]
    print(
        f"labeled={overall['labeled']} good={overall['good']} bad={overall['bad']} "
        f"unlabeled={overall['unlabeled']} precision={overall['precision']:.0%}"
    )
    for group_name, buckets in summary["buckets"].items():
        print(f"\n[{group_name}]")
        for bucket, stats in buckets.items():
            print(
                f"{bucket}: total={stats['total']} labeled={stats['labeled']} "
                f"good={stats['good']} bad={stats['bad']} precision={stats['precision']:.0%}"
            )


def main():
    parser = argparse.ArgumentParser(description="Summarize user feedback on saved opportunities.")
    parser.add_argument(
        "--db",
        default=os.getenv("DEALS_DB_PATH", str(DEFAULT_DEALS_DB_PATH)),
        help="Path to the opportunities SQLite database.",
    )
    args = parser.parse_args()

    summary = summarize_feedback(OpportunityStore(args.db).list_feedback_rows())
    print_summary(summary)


if __name__ == "__main__":
    main()

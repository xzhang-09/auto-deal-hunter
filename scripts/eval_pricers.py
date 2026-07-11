"""
Evaluate PricerAgent's price estimation accuracy on the McAuley holdout sample
produced by build_vector_store.py. The holdout items are excluded from the vector
store, so this measures genuine RAG generalization rather than exact-match lookup.
"""
import argparse
import json
import logging
import os

from dotenv import load_dotenv

from evaluation.pricer import format_summary_line, summarize, threshold_violations
from infra.paths import DEFAULT_EVAL_HOLDOUT_PATH, DEFAULT_VECTORSTORE_PATH

load_dotenv(override=True)

logging.getLogger().setLevel(logging.WARNING)
for _ in ["agents", "chromadb", "httpx", "openai"]:
    logging.getLogger(_).setLevel(logging.WARNING)


def load_holdout(path: str, size: int) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:size]


def get_collection(db_path: str):
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection("products")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PricerAgent (RAG + GPT-4o-mini) on held-out McAuley items."
    )
    parser.add_argument(
        "--size", type=int, default=200, help="Number of holdout samples to evaluate (default: 200)."
    )
    parser.add_argument("--output-json", help="Optional path to write aggregate metrics as JSON.")
    parser.add_argument("--max-mae", type=float, help="Fail when MAE exceeds this value.")
    parser.add_argument("--max-abs-bias", type=float, help="Fail when absolute bias exceeds this value.")
    parser.add_argument("--max-over-rate", type=float, help="Fail when over-prediction rate exceeds this value.")
    args = parser.parse_args()

    holdout_path = os.getenv("EVAL_HOLDOUT_PATH", str(DEFAULT_EVAL_HOLDOUT_PATH))
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"No holdout file at {holdout_path}. Run build_vector_store.py first.")

    items = load_holdout(holdout_path, args.size)
    print(f"Evaluating on {len(items)} held-out items.\n")

    db_path = os.getenv("PRODUCTS_VECTORSTORE_PATH", str(DEFAULT_VECTORSTORE_PATH))
    collection = get_collection(db_path)

    from agents.pricer_agent import PricerAgent

    agent = PricerAgent(collection)

    guesses = []
    truths = []
    failures = 0
    for item in items:
        # The pricer fails loudly (ValueError) when it can't produce a usable estimate, e.g.
        # the model echoes the placeholder because the RAG context is uninformative. One
        # unpriceable item must not kill the whole eval: skip it and report the failure rate,
        # which is itself a quality signal worth tracking across configurations.
        try:
            guess = agent.price(item["summary"])
        except ValueError as exc:
            failures += 1
            print(f"  SKIP (no usable estimate)  {item['title'][:50]}  [{exc}]")
            continue
        truth = item["price"]
        guesses.append(guess)
        truths.append(truth)
        print(f"  guess=${guess:,.2f}  truth=${truth:,.2f}  error=${abs(guess - truth):,.2f}  {item['title'][:50]}")

    stats = summarize(guesses, truths)
    stats["n_failed"] = failures
    if failures:
        print(f"\n{failures} item(s) skipped: pricer produced no usable estimate.")
    print(f"\n{format_summary_line(stats)}")
    from infra import usage

    print(usage.TRACKER.report())
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)

    violations = threshold_violations(
        stats,
        max_mae=args.max_mae,
        max_abs_bias=args.max_abs_bias,
        max_over_rate=args.max_over_rate,
    )
    if violations:
        raise SystemExit("Evaluation thresholds failed: " + "; ".join(violations))


if __name__ == "__main__":
    main()

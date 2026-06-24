"""
Evaluate FrontierAgent's price estimation accuracy on the McAuley holdout sample
produced by build_vector_store.py. The holdout items are excluded from the vector
store, so this measures genuine RAG generalization rather than exact-match lookup.
"""
import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

logging.getLogger().setLevel(logging.WARNING)
for _ in ["agents", "chromadb", "httpx", "openai"]:
    logging.getLogger(_).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.paths import DEFAULT_EVAL_HOLDOUT_PATH, DEFAULT_VECTORSTORE_PATH


def load_holdout(path: str, size: int) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:size]


def get_collection(db_path: str):
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection("products")


def summarize(guesses: list[float], truths: list[float]) -> dict[str, float]:
    """Accuracy plus bias diagnostics against ground-truth prices.

    MAE/RMSE measure raw accuracy; `bias` (mean signed error) and `over_rate` (share of
    estimates above the truth) expose directional bias. A pricer that systematically
    over-predicts is the root cause behind estimates exceeding a deal's list price, so
    these are the metrics to watch when calibrating."""
    n = len(guesses)
    if n == 0:
        raise ValueError("summarize() requires at least one (guess, truth) pair")
    abs_errors = [abs(g - t) for g, t in zip(guesses, truths)]
    signed_errors = [g - t for g, t in zip(guesses, truths)]
    return {
        "n": n,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(sum(e * e for e in abs_errors) / n),
        "bias": sum(signed_errors) / n,
        "over_rate": sum(1 for s in signed_errors if s > 0) / n,
    }


def format_summary_line(stats: dict[str, float]) -> str:
    bias = stats["bias"]
    bias_str = f"{'+' if bias >= 0 else '-'}${abs(bias):,.2f}"
    return (
        f"MAE: ${stats['mae']:,.2f}   RMSE: ${stats['rmse']:,.2f}   "
        f"Bias: {bias_str}   Over-prediction: {stats['over_rate']:.0%}   n={stats['n']}"
    )


def threshold_violations(
    stats: dict[str, float],
    max_mae: float | None = None,
    max_abs_bias: float | None = None,
    max_over_rate: float | None = None,
) -> list[str]:
    violations = []
    if max_mae is not None and stats["mae"] > max_mae:
        violations.append(f"mae {stats['mae']:.2f} > max_mae {max_mae:.2f}")
    if max_abs_bias is not None and abs(stats["bias"]) > max_abs_bias:
        violations.append(
            f"abs(bias) {abs(stats['bias']):.2f} > max_abs_bias {max_abs_bias:.2f}"
        )
    if max_over_rate is not None and stats["over_rate"] > max_over_rate:
        violations.append(
            f"over_rate {stats['over_rate']:.2f} > max_over_rate {max_over_rate:.2f}"
        )
    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FrontierAgent (RAG + GPT-4o-mini) on held-out McAuley items."
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

    from agents.frontier_agent import FrontierAgent

    agent = FrontierAgent(collection)

    guesses = []
    truths = []
    for item in items:
        guess = agent.price(item["summary"])
        truth = item["price"]
        guesses.append(guess)
        truths.append(truth)
        print(f"  guess=${guess:,.2f}  truth=${truth:,.2f}  error=${abs(guess - truth):,.2f}  {item['title'][:50]}")

    stats = summarize(guesses, truths)
    print(f"\n{format_summary_line(stats)}")
    from app import usage

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

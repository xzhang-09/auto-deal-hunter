"""Evaluate the RAG *retriever* in isolation, separately from the LLM pricer.

eval_pricers.py measures end-to-end price accuracy, which blends two failure modes:
a bad retriever (wrong neighbors) and a bad LLM (wrong reasoning over good neighbors).
When MAE is high you can't tell which to fix. This script scores only the retriever on
the held-out items: are the neighbors the right category, and are their prices close to
the held-out item's true price? Cheap (no LLM calls) and diagnostic.

Metrics (averaged over the holdout):
  - category_precision@k : share of the k neighbors in the same category as the query item
  - price_mape@k         : median-neighbor price vs. true price, mean absolute % error
  - hit_rate@k           : share of queries with >=1 same-category neighbor
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EMBEDDING_MODEL
from app.paths import DEFAULT_EVAL_HOLDOUT_PATH, DEFAULT_VECTORSTORE_PATH


def retrieval_metrics(neighbors: list[dict], item: dict) -> dict[str, float]:
    """Per-query retrieval metrics given the retrieved neighbor metadatas and the query item."""
    same_category = [n for n in neighbors if n.get("category") == item["category"]]
    neighbor_prices = [n["price"] for n in neighbors if n.get("price") is not None]
    median_price = statistics.median(neighbor_prices) if neighbor_prices else None
    truth = item["price"]
    price_ape = abs(median_price - truth) / truth if (median_price is not None and truth) else None
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
        "category_precision": sum(m["category_precision"] for m in per_query) / n,
        "hit_rate": sum(m["hit"] for m in per_query) / n,
        "price_mape": (sum(apes) / len(apes)) if apes else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG retriever on held-out McAuley items.")
    parser.add_argument("--size", type=int, default=200, help="Number of holdout items (default: 200).")
    parser.add_argument("--k", type=int, default=5, help="Neighbors to retrieve per query (default: 5).")
    parser.add_argument("--output-json", help="Optional path to write aggregate metrics as JSON.")
    args = parser.parse_args()

    holdout_path = os.getenv("EVAL_HOLDOUT_PATH", str(DEFAULT_EVAL_HOLDOUT_PATH))
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"No holdout file at {holdout_path}. Run build_vector_store.py first.")
    with open(holdout_path) as f:
        items = json.load(f)[: args.size]
    print(f"Evaluating retrieval on {len(items)} held-out items (k={args.k}).\n")

    import chromadb
    from sentence_transformers import SentenceTransformer

    db_path = os.getenv("PRODUCTS_VECTORSTORE_PATH", str(DEFAULT_VECTORSTORE_PATH))
    collection = chromadb.PersistentClient(path=db_path).get_or_create_collection("products")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    per_query = []
    for item in items:
        vector = encoder.encode([item["summary"]]).astype(float).tolist()
        results = collection.query(query_embeddings=vector, n_results=args.k)
        neighbors = results["metadatas"][0]
        per_query.append(retrieval_metrics(neighbors, item))

    stats = aggregate(per_query)
    print(
        f"category_precision@{args.k}: {stats['category_precision']:.0%}   "
        f"hit_rate@{args.k}: {stats['hit_rate']:.0%}   "
        f"price_MAPE@{args.k}: {stats['price_mape']:.0%}   n={stats['n']}"
    )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()

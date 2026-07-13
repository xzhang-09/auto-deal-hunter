"""Evaluate the RAG *retriever* in isolation, separately from the LLM pricer.

eval_pricers.py measures end-to-end price accuracy, which blends two failure modes:
a bad retriever (wrong neighbors) and a bad LLM (wrong reasoning over good neighbors).
When MAE is high you can't tell which to fix. This script scores only the retriever on
the held-out items: are the neighbors the right category, and are their prices close to
the held-out item's true price? Cheap (no LLM calls) and diagnostic.

Metrics (averaged over the holdout):
  - category_precision@k : share of the k neighbors in the same category as the query item
  - price_medianAPE@k    : median over queries of |median-neighbor price - true price| / true price
                           (robust headline; the outlier-sensitive mean APE is also reported)
  - hit_rate@k           : share of queries with >=1 same-category neighbor
"""
import argparse
import json
import os

from dotenv import load_dotenv

from auto_deal_hunter.evaluation.retrieval import aggregate, retrieval_metrics
from auto_deal_hunter.infra.config import EMBEDDING_MODEL
from auto_deal_hunter.infra.paths import DEFAULT_EVAL_HOLDOUT_PATH, DEFAULT_VECTORSTORE_PATH

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG retriever on held-out McAuley items.")
    parser.add_argument("--size", type=int, default=200, help="Number of holdout items (default: 200).")
    parser.add_argument("--k", type=int, default=5, help="Neighbors to retrieve per query (default: 5).")
    parser.add_argument(
        "--rerank",
        choices=["off", "cross-encoder", "llm"],
        default="off",
        help="Optional second-stage reranker to apply before scoring.",
    )
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
    reranker = None
    if args.rerank != "off":
        from auto_deal_hunter.core.reranker import build_reranker
        from auto_deal_hunter.infra.config import RERANK_CANDIDATES

        reranker = build_reranker(args.rerank)
        n_results = max(args.k, RERANK_CANDIDATES)
    else:
        n_results = args.k

    per_query = []
    for item in items:
        # normalize_embeddings to match the build path (unit vectors under cosine distance).
        vector = encoder.encode([item["summary"]], normalize_embeddings=True).astype(float).tolist()
        results = collection.query(query_embeddings=vector, n_results=n_results)
        neighbors = results["metadatas"][0]
        if reranker is not None:
            order = reranker.rerank(item["summary"], results["documents"][0])[: args.k]
            neighbors = [neighbors[i] for i in order]
        per_query.append(retrieval_metrics(neighbors, item))

    stats = aggregate(per_query)
    print(
        f"category_precision@{args.k}: {stats['category_precision']:.0%}   "
        f"hit_rate@{args.k}: {stats['hit_rate']:.0%}   "
        f"price_medianAPE@{args.k}: {stats['price_median_ape']:.0%}   "
        f"(meanAPE: {stats['price_mape']:.0%})   n={stats['n']}"
    )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()

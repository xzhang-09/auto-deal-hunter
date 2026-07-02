"""
Build ChromaDB vector store from McAuley-Lab/Amazon-Reviews-2023 (Electronics category).
Independent of DealNews: gives the LLM real Amazon listing prices to reason from,
instead of letting it guess a price from memorized training data.
Run once before using the agent. Holds out a sample for eval_pricers.py to avoid data leakage.
"""
import json
import os
import random

import chromadb
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from infra.config import EMBEDDING_MODEL, VECTOR_SPACE
from infra.paths import DEFAULT_EVAL_HOLDOUT_PATH, DEFAULT_VECTORSTORE_PATH, ensure_data_dirs
from domain.identity import ItemKind
from domain.item import Item
from ingest.identity import extract_identity_rule

load_dotenv(override=True)

DATASET = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORY = os.getenv("MCAULEY_CATEGORY", "Electronics")
DB_PATH = os.getenv("PRODUCTS_VECTORSTORE_PATH", str(DEFAULT_VECTORSTORE_PATH))
MAX_ITEMS = int(os.getenv("MCAULEY_MAX_ITEMS", "50000"))
HOLDOUT_SIZE = int(os.getenv("EVAL_HOLDOUT_SIZE", "500"))
SEED = 42


def fetch_items() -> list[Item]:
    api = HfApi()
    files = api.list_repo_files(DATASET, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith(f"raw_meta_{CATEGORY}/") and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"No parquet shards found for category {CATEGORY}")
    print(f"Found {len(shards)} shard(s) for {CATEGORY}")

    items = []
    for shard in tqdm(shards, desc="Downloading shards"):
        path = hf_hub_download(DATASET, shard, repo_type="dataset")
        table = pq.read_table(path, columns=["title", "main_category", "price", "description"])
        for row in table.to_pylist():
            item = Item.from_mcauley_row(row)
            if not item:
                continue
            identity = extract_identity_rule(item.summary)
            if identity is not None:
                # Bundles/subscriptions cannot be valued on a single-unit basis, so they make
                # misleading comparables (and bad holdout queries); drop them entirely. Keep
                # multipacks but record the pack size so the query path normalizes to per-unit.
                if identity.kind in (ItemKind.BUNDLE, ItemKind.SUBSCRIPTION):
                    continue
                item.quantity = identity.quantity
                item.variant = identity.variant
            items.append(item)
    print(f"Loaded {len(items)} items with valid price")
    return items


def main():
    ensure_data_dirs()
    items = fetch_items()

    random.seed(SEED)
    random.shuffle(items)

    holdout, items = items[:HOLDOUT_SIZE], items[HOLDOUT_SIZE:]
    if len(items) > MAX_ITEMS:
        items = items[:MAX_ITEMS]
    print(f"Using {len(items)} items for vector store, {len(holdout)} held out for eval")

    with open(DEFAULT_EVAL_HOLDOUT_PATH, "w") as f:
        json.dump([item.model_dump() for item in holdout], f, indent=2)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("products")
    except Exception:
        pass
    # Stamp the embedding model AND the distance metric into the collection so the query path
    # (PricerAgent) can refuse a store built with a different embedder or space. hnsw:space sets
    # cosine distance (mpnet is tuned for cosine; see infra.config.VECTOR_SPACE); without it
    # Chroma defaults to L2, which ranks differently and gives un-thresholdable distances.
    collection = client.create_collection(
        "products",
        metadata={"embedding_model": EMBEDDING_MODEL, "hnsw:space": VECTOR_SPACE},
    )

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    # encode_batch_size stays small: large single batches are ~25x slower on MPS (Apple GPU)
    # than CPU due to backend overhead, even though MPS wins at small batch sizes.
    encode_batch_size = 64
    batch_size = 1000
    for i in tqdm(range(0, len(items), batch_size), desc="Embedding"):
        batch = items[i : i + batch_size]
        documents = [item.summary for item in batch]
        # normalize_embeddings: store unit vectors so cosine == dot product and distances are
        # consistent with the (also-normalized) query path. Must stay in sync with find_similars.
        vectors = (
            encoder.encode(documents, batch_size=encode_batch_size, normalize_embeddings=True)
            .astype(float)
            .tolist()
        )
        metadatas = [
            {
                "category": item.category,
                "price": item.price,
                "quantity": item.quantity,
                "variant": item.variant or "",
            }
            for item in batch
        ]
        ids = [f"doc_{j}" for j in range(i, min(i + batch_size, len(items)))]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

    print(f"Vector store built at {DB_PATH}")


if __name__ == "__main__":
    main()

"""
Build ChromaDB vector store from HuggingFace items dataset.
Uses train split only (excludes val/test) to avoid data leakage during evaluation.
Run once before using the agent. Requires HF_TOKEN in .env.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import login
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from agents.items import Item

load_dotenv(override=True)

DB_PATH = os.getenv("PRODUCTS_VECTORSTORE_PATH", "products_vectorstore")
LITE_MODE = os.getenv("LITE_MODE", "true").lower() == "true"


def main():
    login(token=os.environ.get("HF_TOKEN", ""))
    dataset = os.getenv("ITEMS_DATASET") or ("ed-donner/items_lite" if LITE_MODE else "ed-donner/items_full")
    train, val, test = Item.from_hub(dataset)
    items = train  # Only train, to avoid data leakage when evaluating on test
    print(f"Loaded {len(items)} items (train only, excludes val/test for fair eval)")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("products")

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    batch_size = 1000
    for i in tqdm(range(0, len(items), batch_size)):
        batch = items[i : i + batch_size]
        documents = [item.summary or item.title for item in batch]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in batch]
        ids = [f"doc_{j}" for j in range(i, min(i + batch_size, len(items)))]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

    print(f"Vector store built at {DB_PATH}")


if __name__ == "__main__":
    main()

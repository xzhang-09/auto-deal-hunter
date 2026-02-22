"""
One-shot evaluation of price estimators (Frontier, Specialist, Ensemble).
Uses util.evaluate with pricer-data test set. Requires build_vector_store.py and build_pricer_data.py.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# Suppress agent logging during eval
logging.getLogger().setLevel(logging.WARNING)
for _ in ["agents", "chromadb", "httpx", "openai"]:
    logging.getLogger(_).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))


def _extract_description(datapoint: dict) -> str:
    """Extract product description from pricer-data prompt format."""
    prompt = datapoint.get("prompt", "")
    parts = prompt.split("\n\nPrice is $")
    if len(parts) < 2:
        return prompt
    before = parts[0]
    # Skip "What does this cost to the nearest dollar?"
    idx = before.find("\n\n")
    return before[idx + 2 :].strip() if idx >= 0 else before


def _load_test_data(dataset_name: str):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    test = dataset.get("test") or dataset.get("validation") or dataset["train"]
    # Ensure prompt/completion columns
    if "prompt" not in test.column_names or "completion" not in test.column_names:
        raise ValueError(
            f"Dataset {dataset_name} must have 'prompt' and 'completion' columns. "
            "Run build_pricer_data.py first."
        )
    return test


def _get_collection():
    import chromadb

    db_path = os.getenv("PRODUCTS_VECTORSTORE_PATH", "products_vectorstore")
    if not os.path.isdir(db_path):
        raise FileNotFoundError(
            f"Vector store not found at {db_path}. Run build_vector_store.py first."
        )
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection("products")


def _frontier_predict(collection):
    from agents.frontier_agent import FrontierAgent

    agent = FrontierAgent(collection)

    def predict(datapoint):
        return agent.price(_extract_description(datapoint))

    return predict


def _specialist_predict():
    from agents.specialist_agent import SpecialistAgent

    agent = SpecialistAgent()

    def predict(datapoint):
        return agent.price(_extract_description(datapoint))

    return predict


def _ensemble_predict(collection):
    from agents.ensemble_agent import EnsembleAgent

    agent = EnsembleAgent(collection)

    def predict(datapoint):
        return agent.price(_extract_description(datapoint))

    return predict


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate price estimators on pricer-data test set."
    )
    parser.add_argument(
        "--agents",
        choices=["frontier", "specialist", "ensemble", "all"],
        default="frontier",
        help="Which agent(s) to evaluate. 'all' runs frontier, specialist, ensemble.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Number of test samples (default: 200).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (default: {HF_USER}/pricer-data).",
    )
    args = parser.parse_args()

    hf_user = os.getenv("HF_USER", "ed-donner")
    dataset_name = args.dataset or os.getenv("PRICER_DATASET") or f"{hf_user}/pricer-data"

    print(f"Loading test data from {dataset_name}...")
    test = _load_test_data(dataset_name)
    size = min(args.size, len(test))
    print(f"Evaluating on {size} samples.\n")

    agents_to_run = []
    if args.agents == "all":
        agents_to_run = ["frontier", "specialist", "ensemble"]
    else:
        agents_to_run = [args.agents]

    collection = None
    if "frontier" in agents_to_run or "ensemble" in agents_to_run:
        collection = _get_collection()

    from util import evaluate

    for agent_name in agents_to_run:
        if agent_name == "frontier":
            print("=" * 60)
            print("Evaluating Frontier Agent (RAG + GPT-4o-mini)")
            print("=" * 60)
            pred = _frontier_predict(collection)
            evaluate(pred, test, size=size)

        elif agent_name == "specialist":
            print("=" * 60)
            print("Evaluating Specialist Agent (Fine-tuned on Modal)")
            print("=" * 60)
            try:
                pred = _specialist_predict()
                evaluate(pred, test, size=size)
            except Exception as e:
                print(
                    f"Skipping Specialist: {e}\n"
                    "Ensure modal deploy pricer_service.py and USE_SPECIALIST is set."
                )

        elif agent_name == "ensemble":
            print("=" * 60)
            print("Evaluating Ensemble Agent (Frontier 80% + Specialist 20%)")
            print("=" * 60)
            try:
                pred = _ensemble_predict(collection)
                evaluate(pred, test, size=size)
            except Exception as e:
                print(
                    f"Skipping Ensemble: {e}\n"
                    "Ensure modal deploy pricer_service.py and USE_SPECIALIST is set."
                )


if __name__ == "__main__":
    main()

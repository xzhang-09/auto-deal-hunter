"""
Prepare pricer training data and push to HuggingFace.
Uses items from ed-donner/items_lite or items_full.
Run before fine-tuning. Requires HF_TOKEN.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from tqdm import tqdm

from agents.items import Item

load_dotenv(override=True)

CUTOFF = 110
LITE_MODE = os.getenv("LITE_MODE", "true").lower() == "true"
HF_USER = os.getenv("HF_USER", "ed-donner")
# Push target: always use HF_USER's repo (you can only push to your own). PRICER_DATASET is for train_pricer to pull.
PUSH_DATASET = f"{HF_USER}/pricer-data"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def main():
    login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)
    items_dataset = f"{HF_USER}/items_lite" if LITE_MODE else f"{HF_USER}/items_full"
    train, val, test = Item.from_hub(items_dataset)
    print(f"Loaded {len(train):,} train, {len(val):,} val, {len(test):,} test")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    for item in tqdm(train + val, desc="Train/Val prompts"):
        item.make_prompts(tokenizer, CUTOFF, do_round=True)
    for item in tqdm(test, desc="Test prompts"):
        item.make_prompts(tokenizer, CUTOFF, do_round=False)

    Item.push_prompts_to_hub(PUSH_DATASET, train, val, test)
    print(f"Pushed to {PUSH_DATASET}")


if __name__ == "__main__":
    main()

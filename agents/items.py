from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def count_tokens(self, tokenizer):
        """Count tokens in summary (for CUTOFF analysis)."""
        return len(tokenizer.encode(self.summary or "", add_special_tokens=False))

    def count_prompt_tokens(self, tokenizer):
        """Count tokens in prompt + completion (for sequence length analysis)."""
        full = (self.prompt or "") + (self.completion or "")
        return len(tokenizer.encode(full, add_special_tokens=False))

    def test_prompt(self) -> str:
        """Return prompt without completion for inference."""
        return (self.prompt or "").split(PREFIX)[0] + PREFIX

    def make_prompts(self, tokenizer, max_tokens: int, do_round: bool):
        tokens = tokenizer.encode(self.summary or "", add_special_tokens=False)
        if len(tokens) > max_tokens:
            summary = tokenizer.decode(tokens[:max_tokens]).rstrip()
        else:
            summary = self.summary or ""
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def to_datapoint(self) -> dict:
        p, c = self.prompt or "", self.completion or ""
        return {"prompt": p, "completion": c, "text": p + c}

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list, val: list, test: list
    ):
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)

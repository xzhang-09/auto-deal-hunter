"""
Modal deployment for fine-tuned Qwen 2.5 price predictor.
Deploy: modal deploy pricer_service.py
Requires: modal token, huggingface-secret (HF_TOKEN) in Modal dashboard.
"""
import os
import modal
from modal import Image

app = modal.App("pricer-service")
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = os.getenv("PRICER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
HF_USER = os.getenv("HF_USER", "vvzxxww")
PRICER_RUN = os.getenv("PRICER_RUN", "2026-02-21_03.21.12-lite")
PRICER_REVISION = os.getenv("PRICER_REVISION")  # None = use latest (main)
FINETUNED_MODEL = os.getenv("PRICER_HUB_MODEL") or f"{HF_USER}/price-{PRICER_RUN}"
QUESTION = "What does this cost to the nearest dollar?"
PREFIX = "Price is $"


@app.cls(
    image=image,
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
)
class Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )
        load_kwargs = {"revision": PRICER_REVISION} if PRICER_REVISION else {}
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_MODEL, **load_kwargs
        )

    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(inputs, max_new_tokens=5)
        result = self.tokenizer.decode(outputs[0])
        parts = result.split(PREFIX)
        contents = parts[1] if len(parts) > 1 else ""
        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0.0

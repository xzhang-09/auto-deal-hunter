# -*- coding: utf-8 -*-
"""train_pricer.py

# Fine-tune Qwen 2.5 3B for price prediction (QLoRA)

**Prerequisites**: build_pricer_data.py has pushed pricer-data; add HF_TOKEN to Google Colab Secrets.

## Step 0: Install dependencies + configure
"""

import subprocess
subprocess.run(
    ["pip", "install", "-q", "torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "accelerate"],
    check=True,
)

import os
from datetime import datetime

def _get_secret(name, default=""):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name, default)

HF_TOKEN = _get_secret("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
HF_USER = _get_secret("HF_USER", "vvzxxww") or os.environ.get("HF_USER", "vvzxxww")
PRICER_DATASET = os.environ.get("PRICER_DATASET") or f"{HF_USER}/pricer-data"
BASE_MODEL = os.environ.get("PRICER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LITE_MODE = os.environ.get("LITE_MODE", "true").lower() == "true"
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}" + ("-lite" if LITE_MODE else "")
HUB_MODEL_NAME = f"{HF_USER}/price-{RUN_NAME}"

# LoRA
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
LORA_R = 32 if LITE_MODE else 256
LORA_ALPHA = LORA_R * 2
TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS

# Training: lite=32 batch 1 epoch, full=256 batch 3 epochs
MAX_SEQUENCE_LENGTH = 128
EPOCHS = 1 if LITE_MODE else 3
BATCH_SIZE = 32 if LITE_MODE else 256
SAVE_STEPS = 100 if LITE_MODE else 200
VAL_SIZE = 500 if LITE_MODE else 1000

print(f"HF_USER={HF_USER} | LITE_MODE={LITE_MODE} | Model will be pushed to {HUB_MODEL_NAME}")

"""## Step 1: Load data"""

from datasets import load_dataset
from huggingface_hub import login

login(token=HF_TOKEN, add_to_git_credential=True)

dataset = load_dataset(PRICER_DATASET)
train = dataset["train"]
val = dataset.get("val") or dataset.get("validation")
test = dataset.get("test", train.select(range(min(1000, len(train)))))

if "text" not in train.column_names and "prompt" in train.column_names and "completion" in train.column_names:
    def add_text(examples):
        examples["text"] = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
        return examples
    train = train.map(add_text, batched=True)
    if val is not None:
        val = val.map(add_text, batched=True)

# val from dataset only, never split from train (avoids data leakage)
if val is not None:
    val = val.select(range(min(VAL_SIZE, len(val))))
else:
    vs = int(len(train) * 0.1)
    val = train.select(range(vs))
    train = train.select(range(vs, len(train)))

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

"""## Step 2: Load tokenizer + model"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

print("Model loaded successfully")

"""## Step 3: Create trainer and train"""

from peft import LoraConfig
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.1,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

# per_device_train_batch_size=BATCH_SIZE; Colab may need cap + gradient_accumulation
_per_device = min(BATCH_SIZE, 16)  # T4/fp32 VRAM limit
_grad_accum = max(1, BATCH_SIZE // _per_device)
train_args = SFTConfig(
    output_dir=f"price-{RUN_NAME}",
    run_name=RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=_per_device,
    gradient_accumulation_steps=_grad_accum,
    per_device_eval_batch_size=1,
    optim="paged_adamw_32bit",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    weight_decay=0.001,
    max_grad_norm=0.3,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    save_strategy="steps",
    logging_steps=5 if LITE_MODE else 10,
    max_length=MAX_SEQUENCE_LENGTH,
    dataset_text_field="text",
    completion_only_loss=True,
    group_by_length=True,
    fp16=False,
    bf16=False,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_strategy="every_save",
    report_to="none",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=val,
    peft_config=lora_config,
    args=train_args,
)

print("Starting training...")
trainer.train()

"""## Step 4: Push to HuggingFace"""

trainer.model.push_to_hub(HUB_MODEL_NAME, private=False)
print(f"Done! Model pushed to {HUB_MODEL_NAME}")

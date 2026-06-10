"""DPO training for Gemma3 270M on top of SFT checkpoint.

Uses trl DPOTrainer with conversational message format.
DPOTrainer handles tokenization + chat template internally.

Usage (on rh-h100-01):
    .venv/bin/python train_dpo_gemma3.py
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

# ============================================================================
# Config
# ============================================================================

SFT_MODEL = "model_weights/gemma3-270m/5k-articles-lr5e5/final"
DPO_DATA = "dpo_pairs_train.jsonl"
OUTPUT_DIR = "model_weights/gemma3-270m/dpo"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16
LR = 1e-6
BETA = 5.0
NUM_EPOCHS = 1
LOGGING_STEPS = 1
SAVE_STEPS = 500
N_SAMPLES = 10000  # testing; set to -1 for full dataset


# ============================================================================
# Data
# ============================================================================

def load_dataset(path: str, n_samples: int = -1) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if 0 < n_samples < len(records):
        records = records[:n_samples]
    print(f"Loaded {len(records)} DPO pairs from {path}")
    return Dataset.from_list(records)


# ============================================================================
# Main
# ============================================================================

def main():
    os.environ.setdefault("WANDB_PROJECT", "small-lm-dpo")

    dataset = load_dataset(DPO_DATA, N_SAMPLES)

    run_name = f"dpo-beta{BETA}-{len(dataset)}"

    config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        loss_type="sigmoid",
        max_length=MAX_SEQ_LEN,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=1.0,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        warmup_ratio=0.1,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to="wandb",
        run_name=run_name,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )

    trainer = DPOTrainer(
        model=SFT_MODEL,
        ref_model=None,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    trainer.processing_class.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()

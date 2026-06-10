"""DPO training for Gemma3 270M on top of SFT checkpoint.

Uses trl DPOTrainer with conversational message format.
DPOTrainer handles tokenization + chat template internally.
Supports multi-GPU via torchrun.

Usage (on rh-h100-01):
    # Single GPU
    .venv/bin/python train_dpo_gemma3.py

    # Multi-GPU (8x H100)
    torchrun --nproc_per_node=8 train_dpo_gemma3.py
"""

import json
import os
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import TrainerCallback

# ============================================================================
# Config
# ============================================================================

SFT_MODEL = "model_weights/gemma3-270m/5k-articles-lr5e5/final"
DPO_DATA = "dpo_pairs_train.jsonl"
OUTPUT_DIR = "model_weights/gemma3-270m/dpo-4gpu"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # 4 GPUs × 4 × 4 = 64 effective
LR = 1e-6
BETA = 5.0
NUM_EPOCHS = 10
LOGGING_STEPS = 1
N_SAMPLES = -1  # full dataset

IS_RANK_0 = int(os.environ.get("LOCAL_RANK", 0)) == 0


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
    if IS_RANK_0:
        print(f"Loaded {len(records)} DPO pairs from {path}")
    return Dataset.from_list(records)


# ============================================================================
# Main
# ============================================================================

class ManualWandbCallback(TrainerCallback):
    """Manual wandb logging on rank 0 only."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if IS_RANK_0 and logs:
            metrics = {k: v for k, v in logs.items() if not k.startswith("_")}
            wandb.log(metrics, step=state.global_step)


def main():
    run_name = f"dpo-beta{BETA}-full-10ep-4gpu"

    if IS_RANK_0:
        wandb.init(
            project="small-lm-dpo",
            name=run_name,
            config={
                "beta": BETA,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM_STEPS,
                "n_gpus": int(os.environ.get("WORLD_SIZE", 1)),
                "effective_bs": BATCH_SIZE * GRAD_ACCUM_STEPS * int(os.environ.get("WORLD_SIZE", 1)),
                "n_samples": N_SAMPLES,
                "max_seq_len": MAX_SEQ_LEN,
                "sft_model": SFT_MODEL,
            },
        )

    dataset = load_dataset(DPO_DATA, N_SAMPLES)

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
        save_strategy="epoch",
        save_total_limit=None,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to="none",
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
        callbacks=[ManualWandbCallback()],
    )

    trainer.train()

    if IS_RANK_0:
        final_dir = os.path.join(OUTPUT_DIR, "final")
        trainer.save_model(final_dir)
        trainer.processing_class.save_pretrained(final_dir)
        print(f"Saved to {final_dir}")
        wandb.finish()


if __name__ == "__main__":
    main()

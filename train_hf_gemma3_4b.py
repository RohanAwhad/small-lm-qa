"""Fine-tune Gemma3 4B-it on chunked Wikipedia QA with reasoning.

Purpose: domain-adapt the 4B teacher before using it for GKD distillation.
Same dataset/format as 270M SFT (train_hf_gemma3.py).

Usage (on rh-h100-01, GPUs 2-5):
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 train_hf_gemma3_4b.py
"""
import json
import os
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "google/gemma-3-4b-it"
TRAIN_DATA = "qa_pairs_chunked_train.jsonl"
MAX_SEQ_LEN = 1024
MAX_REASONING_TOKENS = 512
OUTPUT_DIR = "model_weights/gemma3-4b/sft"
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # effective bs = 4 * 4 GPUs * 4 = 64 (with FSDP)
LR = 5e-5
NUM_EPOCHS = 1
LOGGING_STEPS = 1
SAVE_STEPS = 200

SYSTEM_PROMPT = "Answer the question using the provided context."

# ============================================================================
# Data
# ============================================================================

def build_dataset(tokenizer) -> Dataset:
    pairs = []
    with open(TRAIN_DATA) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} QA pairs from {TRAIN_DATA}")

    texts = []
    n_skipped_reasoning = 0
    n_skipped_length = 0
    for p in pairs:
        reasoning = p.get("reasoning_content", "")
        rc_toks = len(tokenizer.encode(reasoning, add_special_tokens=False))
        if rc_toks > MAX_REASONING_TOKENS:
            n_skipped_reasoning += 1
            continue

        chunks_text = "\n\n".join(p["context_chunks"])
        assistant_content = f"<reasoning>\n{reasoning}\n</reasoning>\n\n{p['regen_answer']}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{chunks_text}\n\nQuestion: {p['question']}"},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        tok_len = len(tokenizer.encode(text, add_special_tokens=False))
        if tok_len > MAX_SEQ_LEN:
            n_skipped_length += 1
            continue

        texts.append(text)

    print(f"Formatted {len(texts)} examples")
    print(f"Skipped: {n_skipped_reasoning} (reasoning > {MAX_REASONING_TOKENS} tok), {n_skipped_length} (seq > {MAX_SEQ_LEN} tok)")
    return Dataset.from_dict({"text": texts})


def tokenize_fn(examples, tokenizer):
    out = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    out["labels"] = [list(ids) for ids in out["input_ids"]]
    return out


@dataclass
class DynamicPadCollator:
    """Pads each batch to the longest sequence in the batch, not max_seq_len."""
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

# ============================================================================
# Main
# ============================================================================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    dataset = build_dataset(tokenizer)
    tokenized = dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    print(f"Tokenized dataset: {len(tokenized)} examples")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        adam_epsilon=1e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        report_to="wandb",
        run_name="gemma3-4b-sft-teacher",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        fsdp="full_shard",
        fsdp_config={
            "forward_prefetch": True,
            "backward_prefetch": "backward_pre",
            "transformer_layer_cls_to_wrap": "Gemma3DecoderLayer",
        },
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=DynamicPadCollator(tokenizer),
    )

    trainer.train()
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()

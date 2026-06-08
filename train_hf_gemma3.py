"""Fine-tune Gemma3 270M on chunked Wikipedia QA with reasoning."""
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "unsloth/gemma-3-270m-it"
TRAIN_DATA = "qa_pairs_chunked.jsonl"
MAX_SEQ_LEN = 1024
MAX_REASONING_TOKENS = 512
OUTPUT_DIR = "model_weights/gemma3-270m/hf_ckpts"
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16
LR = 1e-6
NUM_EPOCHS = 10
MAX_STEPS = -1  # full epoch
LOGGING_STEPS = 1
SAVE_STEPS = 500

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
        # Filter: skip if reasoning exceeds token budget
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

        # Filter: skip if total sequence exceeds max
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
        padding="max_length",
    )
    # mask padding tokens in labels with -100 so they're ignored in loss
    labels = []
    for ids, mask in zip(out["input_ids"], out["attention_mask"]):
        labels.append([id if m == 1 else -100 for id, m in zip(ids, mask)])
    out["labels"] = labels
    return out

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
        attn_implementation="eager",
        device_map="auto",
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()

"""Fine-tune Gemma3 270M on Wikipedia QA pairs using HF Transformers."""
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from utils.wikipedia_loader import load_articles_by_id

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "unsloth/gemma-3-270m-it"
TRAIN_DATA = "qa_train.json"
MAX_SEQ_LEN = 4096
OUTPUT_DIR = "model_weights/gemma3-270m/hf_ckpts"
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LR = 1e-5
NUM_EPOCHS = 1
MAX_STEPS = 10  # set to -1 for full epoch
LOGGING_STEPS = 1
SAVE_STEPS = 50

# ============================================================================
# Data
# ============================================================================

def build_dataset(tokenizer) -> Dataset:
    pairs = json.load(open(TRAIN_DATA))
    print(f"Loaded {len(pairs)} QA pairs from {TRAIN_DATA}")

    article_ids = {p["article_id"] for p in pairs}
    articles = load_articles_by_id(article_ids)
    print(f"Loaded {len(articles)} articles")

    texts = []
    for p in pairs:
        ctx = articles.get(p["article_id"], "")
        if not ctx:
            continue
        # truncate context
        max_ctx_chars = MAX_SEQ_LEN * 8
        ctx = ctx[:max_ctx_chars]

        messages = [
            {"role": "user", "content": f"Article:\n{ctx}\n\nQuestion: {p['question']}\n\nAnswer:"},
            {"role": "assistant", "content": p["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    print(f"Formatted {len(texts)} examples")
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
        warmup_steps=10,
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
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print(f"Saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()

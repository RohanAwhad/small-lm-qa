"""Fine-tune Gemma3 270M on Wikipedia QA pairs."""
import json
import os
import random
import time

from tinygrad import Tensor, nn
from tinygrad.nn.optim import AdamW
from transformers import AutoTokenizer

from gemma3 import (
    Gemma3, Gemma3Config, cross_entropy_loss, train_step_grad_accum,
    load_pretrained, save_checkpoint, PRETRAINED_PATH, CKPT_DIR,
)
from utils.wikipedia_loader import load_articles_by_id

# ============================================================================
# Config
# ============================================================================

TOKENIZER_PATH = "model_weights/gemma3-270m/pretrained"
TRAIN_DATA = "qa_train.json"
MAX_SEQ_LEN = 256
MICRO_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16  # effective batch size = 16
LR = 1e-5
NUM_STEPS = 10
CKPT_EVERY = 50

SYSTEM_PROMPT = "You are a helpful assistant. Answer the question based only on the provided article text."

# ============================================================================
# Data
# ============================================================================

def format_qa(question: str, answer: str, context: str, tokenizer) -> list[int]:
    """Format a QA pair into chat-style token ids."""
    # truncate context to fit — leave room for question + answer + template
    max_ctx_chars = MAX_SEQ_LEN * 8  # rough char budget (~4 chars/token, 2x safety)
    context = context[:max_ctx_chars]

    messages = [
        {"role": "user", "content": f"Article:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        {"role": "assistant", "content": answer},
    ]
    # use chat template if available, otherwise manual format
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:MAX_SEQ_LEN]


def load_dataset(tokenizer) -> list[list[int]]:
    """Load and tokenize all QA pairs."""
    pairs = json.load(open(TRAIN_DATA))
    print(f"Loaded {len(pairs)} QA pairs from {TRAIN_DATA}")

    # load article texts
    article_ids = {p["article_id"] for p in pairs}
    articles = load_articles_by_id(article_ids)
    print(f"Loaded {len(articles)} articles")

    tokenized = []
    skipped = 0
    for p in pairs:
        ctx = articles.get(p["article_id"], "")
        if not ctx:
            skipped += 1
            continue
        ids = format_qa(p["question"], p["answer"], ctx, tokenizer)
        if len(ids) >= 8:  # skip very short sequences
            tokenized.append(ids)

    print(f"Tokenized {len(tokenized)} examples (skipped {skipped})")
    return tokenized


def make_microbatches(dataset: list[list[int]], num: int, micro_bs: int) -> list[Tensor]:
    """Sample `num` microbatches of size `micro_bs`, each padded to max len."""
    batches = []
    for _ in range(num):
        samples = random.choices(dataset, k=micro_bs)
        max_len = min(max(len(s) for s in samples), MAX_SEQ_LEN)
        padded = [s[:max_len] + [0] * (max_len - len(s)) for s in samples]
        batches.append(Tensor(padded))
    return batches

# ============================================================================
# Main
# ============================================================================

def main():
    Tensor.training = True

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # data
    dataset = load_dataset(tokenizer)

    # model
    config = Gemma3Config()
    model = Gemma3(config)
    load_pretrained(model, PRETRAINED_PATH)

    params = nn.state.get_parameters(model)
    optimizer = AdamW(params, lr=LR)
    total = sum(p.numel() for p in params)
    eff_bs = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
    print(f"\nGemma3 270M — {total:,} params ({total/1e6:.1f}M)")
    print(f"Training: {NUM_STEPS} steps, micro_bs={MICRO_BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}, eff_bs={eff_bs}, seq_len={MAX_SEQ_LEN}, lr={LR}")

    # train
    os.makedirs(CKPT_DIR, exist_ok=True)
    print()
    for step in range(NUM_STEPS):
        microbatches = make_microbatches(dataset, GRAD_ACCUM_STEPS, MICRO_BATCH_SIZE)
        t0 = time.time()
        avg_loss = train_step_grad_accum(model, optimizer, microbatches, GRAD_ACCUM_STEPS)
        dt = time.time() - t0
        print(f"step {step:4d}: loss={avg_loss:.4f}  ({dt:.1f}s)")

        if CKPT_EVERY and (step + 1) % CKPT_EVERY == 0:
            save_checkpoint(model, os.path.join(CKPT_DIR, f"step_{step+1}.safetensors"))


if __name__ == "__main__":
    main()

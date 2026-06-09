"""Generate DPO rollouts: sample N responses per question.

Two modes:
  --hf:  Direct HF Transformers inference on GPU (no server needed)
  default: Async OpenAI client against a vLLM server

Output: JSONL where each line has the original fields + rollouts list.
Resume support: skips article_id::question keys already in the output file.

Usage:
    # HF mode (recommended for small models — no server setup needed)
    .venv/bin/python generate_dpo_rollouts.py qa_pairs_chunked_train.jsonl \
        -o dpo_rollouts.jsonl --hf -m model_weights/gemma3-270m/hf_ckpts/checkpoint-500

    # vLLM server mode
    .venv/bin/python generate_dpo_rollouts.py qa_pairs_chunked_train.jsonl \
        -o dpo_rollouts.jsonl --base-url http://localhost:8001/v1
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

N_ROLLOUTS = 5
TEMPERATURE = 1.0
MAX_TOKENS = 1024
MAX_MODEL_LEN = 2048
TOKEN_BUFFER = 100  # headroom for chat template overhead
MIN_OUTPUT_TOKENS = 64  # skip if less room than this
MAX_CONCURRENT = 160
LOG_DIR = Path("logs")
SYSTEM_PROMPT = "Answer the question using the provided context."

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "generate_dpo_rollouts.log", level=log_level, rotation="10 MB")


def make_key(article_id: int, question: str) -> str:
    return f"{article_id}::{question}"


def load_done_keys(output_path: Path) -> set[str]:
    """Load keys already in the output file for resume support."""
    if not output_path.exists():
        return set()
    keys = set()
    for line in output_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        keys.add(make_key(r["article_id"], r["question"]))
    logger.info(f"Resume: {len(keys)} questions already done in {output_path}")
    return keys


def build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~3.5 chars per token for English text."""
    total_chars = sum(len(m["content"]) for m in messages)
    return int(total_chars / 3.5)


def compute_max_tokens(prompt_tokens: int) -> int:
    """Dynamic max_tokens: fit within MAX_MODEL_LEN with buffer."""
    available = MAX_MODEL_LEN - prompt_tokens - TOKEN_BUFFER
    return min(available, MAX_TOKENS)


# ============================================================================
# HF Transformers mode (direct GPU inference, no server)
# ============================================================================

def main_hf(input_path: Path, output_path: Path, model_id: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load input
    pairs = [json.loads(l) for l in input_path.read_text().strip().splitlines() if l.strip()]
    logger.info(f"Loaded {len(pairs)} QA pairs from {input_path}")

    # Resume support
    done_keys = load_done_keys(output_path)
    todo = [p for p in pairs if make_key(p["article_id"], p["question"]) not in done_keys]
    logger.info(f"TODO: {len(todo)} questions ({len(pairs) - len(todo)} already done)")

    if not todo:
        logger.info("Nothing to do — all questions already processed")
        return

    # Load model
    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    logger.info(f"Model loaded on {model.device}")

    t0 = time.monotonic()
    written = 0
    with open(output_path, "a") as f:
        for pair in todo:
            context = "\n\n".join(pair["context_chunks"])
            messages = build_messages(pair["question"], context)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=getattr(model.config, "max_position_embeddings", 8192) - MAX_TOKENS,
            ).to(model.device)

            rollouts = []
            for _ in range(N_ROLLOUTS):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_k=0,
                    )
                input_len = inputs["input_ids"].shape[1]
                generated = outputs[0][input_len:]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                rollouts.append(text)

            record = {
                "article_id": pair["article_id"],
                "title": pair["title"],
                "question": pair["question"],
                "golden_answer": pair["golden_answer"],
                "context": context,
                "rollouts": rollouts,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written % 100 == 0:
                elapsed = time.monotonic() - t0
                logger.info(f"Progress: {written}/{len(todo)} done ({elapsed:.0f}s, "
                            f"{written / elapsed:.1f} q/s)")

    elapsed = time.monotonic() - t0
    logger.info(f"Done. {written} written in {elapsed:.1f}s ({written / max(elapsed, 0.1):.1f} q/s)")


# ============================================================================
# vLLM server mode (async OpenAI client)
# ============================================================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=10, jitter=2),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: logger.warning(
        f"Rollout retry {rs.attempt_number}/5: {rs.outcome.exception()!r}"
    ),
    reraise=True,
)
async def generate_one_vllm(
    client: "AsyncOpenAI",
    semaphore: asyncio.Semaphore,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> list[str]:
    """Generate N rollouts for a single prompt with retry."""
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            n=N_ROLLOUTS,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
        )
    return [choice.message.content or "" for choice in resp.choices]


async def main_vllm(input_path: Path, output_path: Path, base_url: str, model: str) -> None:
    from openai import AsyncOpenAI

    # Load input
    pairs = [json.loads(l) for l in input_path.read_text().strip().splitlines() if l.strip()]
    logger.info(f"Loaded {len(pairs)} QA pairs from {input_path}")

    # Resume support
    done_keys = load_done_keys(output_path)
    todo = [p for p in pairs if make_key(p["article_id"], p["question"]) not in done_keys]
    logger.info(f"TODO: {len(todo)} questions ({len(pairs) - len(todo)} already done)")

    if not todo:
        logger.info("Nothing to do — all questions already processed")
        return

    client = AsyncOpenAI(api_key="dummy", base_url=base_url)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    f = open(output_path, "a")
    written = 0
    failed = 0
    skipped = 0
    lock = asyncio.Lock()

    async def process_one(pair: dict) -> None:
        nonlocal written, failed, skipped
        context = "\n\n".join(pair["context_chunks"])
        messages = build_messages(pair["question"], context)

        prompt_tokens = estimate_prompt_tokens(messages)
        max_tokens = compute_max_tokens(prompt_tokens)
        if max_tokens < MIN_OUTPUT_TOKENS:
            async with lock:
                skipped += 1
            logger.debug(f"Skipped article_id={pair['article_id']} — prompt too long "
                         f"(~{prompt_tokens} tokens, only {max_tokens} left)")
            return

        rollouts = await generate_one_vllm(client, semaphore, model, messages, max_tokens)
        record = {
            "article_id": pair["article_id"],
            "title": pair["title"],
            "question": pair["question"],
            "golden_answer": pair["golden_answer"],
            "context": context,
            "rollouts": rollouts,
        }
        async with lock:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written % 500 == 0:
                logger.info(f"Progress: {written}/{len(todo)} done, {skipped} skipped, {failed} failed")

    t0 = time.monotonic()
    results = await asyncio.gather(
        *[process_one(p) for p in todo], return_exceptions=True,
    )
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed += 1
            logger.error(f"Failed article_id={todo[i]['article_id']} "
                         f"q='{todo[i]['question'][:60]}': {result}")

    f.close()
    elapsed = time.monotonic() - t0
    logger.info(
        f"Done. {written} written, {skipped} skipped, {failed} failed in {elapsed:.1f}s "
        f"({written / max(elapsed, 0.1):.1f} q/s)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO rollouts")
    parser.add_argument("input", nargs="?", default="qa_pairs_chunked_train.jsonl",
                        help="Input JSONL with QA pairs")
    parser.add_argument("-o", "--output", default="dpo_rollouts.jsonl",
                        help="Output JSONL with rollouts")
    parser.add_argument("-m", "--model", default="model_weights/gemma3-270m/hf_ckpts/checkpoint-500",
                        help="HF model ID or local checkpoint path")
    parser.add_argument("--hf", action="store_true",
                        help="Use HF Transformers directly instead of vLLM server")
    parser.add_argument("--base-url", default="http://localhost:8001/v1",
                        help="vLLM server base URL (ignored with --hf)")
    args = parser.parse_args()

    if args.hf:
        main_hf(Path(args.input), Path(args.output), args.model)
    else:
        asyncio.run(main_vllm(Path(args.input), Path(args.output), args.base_url, args.model))

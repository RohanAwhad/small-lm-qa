"""Generate DPO rollouts: sample N responses per question via vLLM server.

Reads qa_pairs_chunked_train.jsonl, sends each prompt to a vLLM-served model
with n=5 and temperature=1.0, collects all completions.

Output: JSONL where each line has the original fields + rollouts list.
Resume support: skips article_id::question keys already in the output file.

Usage:
    # Against vLLM-served Gemma3 270M on node 01
    .venv/bin/python generate_dpo_rollouts.py qa_pairs_chunked_train.jsonl \
        -o dpo_rollouts.jsonl \
        --base-url http://localhost:8001/v1 \
        --model model_weights/gemma3-270m/hf_ckpts/checkpoint-500
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

N_ROLLOUTS = 5
TEMPERATURE = 1.0
MAX_TOKENS = 1024
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


async def main(input_path: Path, output_path: Path, base_url: str, model: str) -> None:
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

    # Shared file handle for streaming writes
    f = open(output_path, "a")
    written = 0
    failed = 0
    lock = asyncio.Lock()

    async def process_one(pair: dict) -> None:
        nonlocal written, failed
        context = "\n\n".join(pair["context_chunks"])
        messages = build_messages(pair["question"], context)

        async with semaphore:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                n=N_ROLLOUTS,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

        rollouts = [choice.message.content or "" for choice in resp.choices]

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
                logger.info(f"Progress: {written}/{len(todo)} done, {failed} failed")

    # Process all
    t0 = time.monotonic()
    tasks = [process_one(p) for p in todo]

    # Use gather with return_exceptions to not crash on individual failures
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed += 1
            logger.error(f"Failed article_id={todo[i]['article_id']}: {result}")

    f.close()
    elapsed = time.monotonic() - t0
    logger.info(
        f"Done. {written} written, {failed} failed in {elapsed:.1f}s "
        f"({len(todo) / elapsed:.1f} q/s)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO rollouts via vLLM server")
    parser.add_argument("input", nargs="?", default="qa_pairs_chunked_train.jsonl",
                        help="Input JSONL with QA pairs")
    parser.add_argument("-o", "--output", default="dpo_rollouts.jsonl",
                        help="Output JSONL with rollouts")
    parser.add_argument("--base-url", default="http://localhost:8001/v1",
                        help="vLLM server base URL")
    parser.add_argument("--model", default="model_weights/gemma3-270m/hf_ckpts/checkpoint-500",
                        help="Model name for vLLM API")
    args = parser.parse_args()

    asyncio.run(main(Path(args.input), Path(args.output), args.base_url, args.model))

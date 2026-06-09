"""Generate answers via vLLM OpenAI-compatible API (async, semaphore-gated)."""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "generate_vllm_answers.log", level="DEBUG", rotation="10 MB")

SYSTEM_PROMPT = "Answer the question using the provided context."
MAX_CONCURRENT = 32
MAX_NEW_TOKENS = 1024


async def generate_answer(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    question: str,
    context: str,
) -> str:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0,
        )
        return resp.choices[0].message.content


async def main(input_path: Path, output_path: Path, base_url: str, model: str):
    raw = input_path.read_text()
    pairs = [json.loads(l) for l in raw.splitlines() if l.strip()]
    logger.info(f"Loaded {len(pairs)} QA pairs from {input_path}")

    has_chunks = "context_chunks" in pairs[0] if pairs else False
    if has_chunks:
        logger.info("Detected chunked input — using context_chunks")
    else:
        logger.info("No context_chunks — using full article text")
        from utils.wikipedia_loader import load_articles_by_id
        article_ids = {p["article_id"] for p in pairs}
        article_texts = load_articles_by_id(article_ids)

    client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    model_label = model.split("/")[-1]

    # Build tasks
    valid_pairs = []
    contexts = []
    for pair in pairs:
        if has_chunks:
            ctx = "\n\n".join(pair["context_chunks"])
        else:
            ctx = article_texts.get(pair["article_id"], "")
            if not ctx:
                logger.warning(f"[{pair['title']}] article not found, skipping")
                continue
        valid_pairs.append(pair)
        contexts.append(ctx)

    logger.info(f"Generating {len(valid_pairs)} answers with concurrency={MAX_CONCURRENT}")
    t0 = time.monotonic()

    async def process_one(idx: int):
        pair = valid_pairs[idx]
        answer = await generate_answer(client, semaphore, model, pair["question"], contexts[idx])
        return idx, answer

    results = [None] * len(valid_pairs)
    done_count = 0
    coros = [process_one(i) for i in range(len(valid_pairs))]

    for coro in asyncio.as_completed(coros):
        idx, answer = await coro
        results[idx] = answer
        done_count += 1
        if done_count % 50 == 0 or done_count == len(valid_pairs):
            logger.info(f"Progress: {done_count}/{len(valid_pairs)} ({time.monotonic()-t0:.1f}s)")

    # Write in original order
    with open(output_path, "w") as f:
        for pair, answer in zip(valid_pairs, results):
            out = {**pair, "model_answer": answer, "model": model_label}
            if "golden_answer" in out and "answer" not in out:
                out["answer"] = out["golden_answer"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - t0
    logger.info(f"Done. {len(valid_pairs)} answers written to {output_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers via vLLM OpenAI API")
    parser.add_argument("input", help="Input JSON/JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM base URL")
    parser.add_argument("-m", "--model", default="google/gemma-3-27b-it", help="Model name")
    args = parser.parse_args()

    asyncio.run(main(Path(args.input), Path(args.output), args.base_url, args.model))

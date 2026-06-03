"""Generate Gemma3 answers for existing QA pairs using article text as context."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from loguru import logger

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:270m"
MAX_CONCURRENT = 4
LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "generate_gemma.log", level="DEBUG", rotation="10 MB")

SYSTEM_PROMPT = "You are a helpful assistant. Answer the question based only on the provided article text."

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"
HF_MAX_PER_REQUEST = 100


async def fetch_articles_by_id(article_ids: set[int]) -> dict[int, str]:
    texts: dict[int, str] = {}
    async with httpx.AsyncClient(timeout=30) as http:
        for offset in range(0, max(article_ids) + 1, HF_MAX_PER_REQUEST):
            ids_in_range = {i for i in article_ids if offset <= i < offset + HF_MAX_PER_REQUEST}
            if not ids_in_range:
                continue
            resp = await http.get(
                HF_ROWS_URL,
                params={"dataset": HF_DATASET, "config": HF_CONFIG, "split": "train", "offset": offset, "length": HF_MAX_PER_REQUEST},
            )
            resp.raise_for_status()
            for r in resp.json()["rows"]:
                if r["row_idx"] in article_ids:
                    texts[r["row_idx"]] = r["row"]["text"]
            logger.info(f"Fetched {len(texts)}/{len(article_ids)} articles")
    return texts


async def generate_answer(
    http: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    question: str,
    context: str,
    title: str,
) -> str | None:
    async with semaphore:
        logger.debug(f"[{title}] generating answer for Q: {question[:60]}...")
        t0 = time.monotonic()

        resp = await http.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Article:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
                ],
                "stream": False,
                "options": {"num_predict": 1024},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["message"]["content"].strip()

        elapsed = time.monotonic() - t0
        logger.info(f"[{title}] answer ({len(answer)} chars) in {elapsed:.1f}s")
        return answer


def checkpoint(output_path: Path, results: list[dict], n: int) -> None:
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    logger.info(f"Checkpoint: {n} answers written to {output_path}")


async def main(input_path: Path, output_path: Path) -> None:
    raw = input_path.read_text().strip()
    if raw.startswith("["):
        pairs = json.loads(raw)
    else:
        pairs = [json.loads(l) for l in raw.splitlines() if l.strip()]
    logger.info(f"Loaded {len(pairs)} QA pairs from {input_path}")

    article_ids = {p["article_id"] for p in pairs}
    logger.info(f"Fetching {len(article_ids)} unique articles...")
    article_texts = await fetch_articles_by_id(article_ids)
    logger.info(f"Fetched {len(article_texts)} article texts")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    output_path.write_text("")

    results: list[dict] = []
    seen = 0

    async def process_pair(pair: dict) -> dict | None:
        nonlocal results, seen
        text = article_texts.get(pair["article_id"], "")
        if not text:
            logger.warning(f"[{pair['title']}] article_id={pair['article_id']} not found, skipping")
            return None
        answer = await generate_answer(http, semaphore, pair["question"], text, pair["title"])
        if answer is None:
            return None
        result = {**pair, "model_answer": answer, "model": OLLAMA_MODEL}
        results.append(result)
        seen += 1
        if seen % 50 == 0:
            checkpoint(output_path, results, seen)
        return result

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=300) as http:
        tasks = [process_pair(p) for p in pairs]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in raw if isinstance(r, Exception)]
        for e in errors:
            logger.error(f"Task failed: {e}")

    checkpoint(output_path, results, len(pairs))
    logger.info(f"Done. {len(results)}/{len(pairs)} answers ({len(errors)} errors) in {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gemma3 answers for QA pairs")
    parser.add_argument("input", nargs="?", default="qa_pairs.jsonl", help="Input JSON/JSONL file (default: qa_pairs.jsonl)")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file (default: <input_stem>_gemma.jsonl)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_gemma").with_suffix(".jsonl")
    asyncio.run(main(in_path, out_path))

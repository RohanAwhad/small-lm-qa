"""Download all Wikipedia English articles from HF datasets server API."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

# --- Config ---
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"
HF_MAX_PER_REQUEST = 100
MAX_CONCURRENT = 10
CHUNK_SIZE = 50  # batches per chunk (50 × 100 = 5,000 articles)

OUTPUT_FILE = Path("wikipedia_en.jsonl")
LOG_DIR = Path("logs")

# --- Logging setup ---
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(
    LOG_DIR / "download_wikipedia.log",
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    rotation="100 MB",
)


async def get_total_articles(http: httpx.AsyncClient) -> int:
    resp = await http.get(
        HF_ROWS_URL,
        params={"dataset": HF_DATASET, "config": HF_CONFIG, "split": "train", "offset": 0, "length": 0},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["num_rows_total"]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    before_sleep=before_sleep_log(logger, "WARNING"),
    reraise=True,
)
async def fetch_batch(http: httpx.AsyncClient, offset: int, length: int) -> list[dict]:
    resp = await http.get(
        HF_ROWS_URL,
        params={"dataset": HF_DATASET, "config": HF_CONFIG, "split": "train", "offset": offset, "length": length},
        timeout=120,
    )
    resp.raise_for_status()
    rows = resp.json()["rows"]
    return [
        {
            "article_id": r["row_idx"],
            "id": r["row"]["id"],
            "url": r["row"]["url"],
            "title": r["row"]["title"],
            "text": r["row"]["text"],
        }
        for r in rows
    ]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def write_jsonl(articles: list[dict], path: Path) -> None:
    with open(path, "a") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")


async def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=MAX_CONCURRENT + 5)) as http:
        dataset_total = await get_total_articles(http)
        total = min(limit, dataset_total) if limit else dataset_total
        logger.info(f"Total articles: {total:,} (dataset: {dataset_total:,}, {total}/{HF_MAX_PER_REQUEST} = {total // HF_MAX_PER_REQUEST} batches)")

        downloaded = count_lines(OUTPUT_FILE)
        remainder = total - downloaded

        if remainder <= 0:
            logger.info(f"All {downloaded:,} articles already downloaded.")
            return

        logger.info(
            f"Resuming from article {downloaded:,} ({remainder:,} remaining, {downloaded / total * 100:.1f}% complete)"
        )

        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def fetch_one(offset: int) -> list[dict]:
            async with sem:
                length = min(HF_MAX_PER_REQUEST, total - offset)
                return await fetch_batch(http, offset, length)

        # Generate remaining offsets
        offsets = list(range(downloaded, total, HF_MAX_PER_REQUEST))
        t0 = time.monotonic()
        last_log = t0
        start_count = downloaded

        # Process in chunks for ordered writes + resume at chunk boundaries
        for chunk_idx, chunk_start in enumerate(range(0, len(offsets), CHUNK_SIZE)):
            chunk_offsets = offsets[chunk_start : chunk_start + CHUNK_SIZE]
            chunk_t0 = time.monotonic()

            tasks = [fetch_one(offset) for offset in chunk_offsets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            articles_in_chunk = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed offset {chunk_offsets[i]} after 5 retries: {type(result).__name__}: {result}")
                    break  # Stop at first gap to keep output sequential for resume
                write_jsonl(result, OUTPUT_FILE)
                downloaded += len(result)
                articles_in_chunk += len(result)

            chunk_elapsed = time.monotonic() - chunk_t0
            logger.debug(
                f"Chunk {chunk_idx + 1}: {articles_in_chunk} articles in {chunk_elapsed:.1f}s "
                f"({articles_in_chunk / chunk_elapsed:.1f}/s)"
            )

            now = time.monotonic()
            if now - last_log > 30:
                elapsed = now - t0
                done = downloaded - start_count
                rate = done / elapsed if elapsed > 0 else 0
                remaining = total - downloaded
                eta = (remaining / rate) if rate > 0 else 0
                logger.info(
                    f"Progress: {downloaded:,}/{total:,} ({downloaded / total * 100:.2f}%) | "
                    f"{rate:.1f} a/s | ETA: {eta / 3600:.1f}h"
                )
                last_log = now

        elapsed = time.monotonic() - t0
        logger.info(
            f"Done. {downloaded:,}/{total:,} articles -> {OUTPUT_FILE} ({elapsed / 3600:.1f}h)"
            f"{' (INCOMPLETE)' if downloaded < total else ''}"
        )


if __name__ == "__main__":
    asyncio.run(main())

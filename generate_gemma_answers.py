"""Generate Gemma3 answers for existing QA pairs using article text as context."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

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


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.ReadTimeout):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
async def _call_ollama(http: httpx.AsyncClient, title: str, question: str, context: str) -> str:
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
    return resp.json()["message"]["content"].strip()


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
        try:
            answer = await _call_ollama(http, title, question, context)
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(f"[{title}] failed after retries: {type(e).__name__}: {e} ({elapsed:.1f}s)")
            return None
        elapsed = time.monotonic() - t0
        logger.info(f"[{title}] answer ({len(answer)} chars) in {elapsed:.1f}s")
        return answer


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

    async def process_pair(pair: dict) -> dict | None:
        text = article_texts.get(pair["article_id"], "")
        if not text:
            logger.warning(f"[{pair['title']}] article_id={pair['article_id']} not found, skipping")
            return None
        answer = await generate_answer(http, semaphore, pair["question"], text, pair["title"])
        if answer is None:
            return None
        return {**pair, "model_answer": answer, "model": OLLAMA_MODEL}

    t0 = time.monotonic()
    written = 0
    total = len(pairs)
    async with httpx.AsyncClient(timeout=120) as http:
        tasks = [process_pair(p) for p in pairs]
        with open(output_path, "w") as f:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    written += 1
                    if written % 10 == 0:
                        logger.info(f"Progress: {written}/{total} ({time.monotonic()-t0:.1f}s)")

    logger.info(f"Done. {written}/{total} answers written to {output_path} in {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gemma3 answers for QA pairs")
    parser.add_argument("input", nargs="?", default="qa_pairs.jsonl", help="Input JSON/JSONL file (default: qa_pairs.jsonl)")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file (default: <input_stem>_gemma.jsonl)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_gemma").with_suffix(".jsonl")
    asyncio.run(main(in_path, out_path))

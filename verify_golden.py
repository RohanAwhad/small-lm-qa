"""Verify golden QA pairs against Wikipedia articles using 4-vote LLM judge.

For each QA pair, 4 independent DeepSeek calls judge whether the answer is
factually correct given the Wikipedia article. Unanimous agreement required.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_CONCURRENT = 50
VOTE_COUNT = 4
LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "verify_golden.log", level=log_level, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}")

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"
HF_MAX_PER_REQUEST = 100


class Verdict(BaseModel):
    reasoning: str = ""
    correct: bool


VERIFY_SYSTEM = "You verify whether a QA pair's answer is factually correct based on a Wikipedia article. Respond only in json format."

VERIFY_USER = """## Wikipedia Article
{article}

## Question
{question}

## Answer
{answer}

Is this answer factually correct based on the article? Check key facts against the article text.
Respond in JSON format: {{"reasoning": "explain which facts are correct or incorrect", "correct": true/false}}"""


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)


def load_json_or_jsonl(path: Path) -> list[dict]:
    raw = path.read_text().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


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


async def single_verdict(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    article_text: str,
    question: str,
    answer: str,
) -> Verdict | None:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": VERIFY_USER.format(article=article_text, question=question, answer=answer)},
            ],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        if not raw:
            return None

        return Verdict.model_validate(json.loads(raw))


async def verify_pair(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pair: dict,
    article_text: str,
) -> dict | None:
    question = pair["question"]
    answer = pair["answer"]
    t0 = time.monotonic()

    coros = [single_verdict(client, semaphore, article_text, question, answer) for _ in range(VOTE_COUNT)]
    results = await asyncio.gather(*coros, return_exceptions=True)

    verdicts: list[dict] = []
    correct_votes = 0
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"[{pair['title']}] vote error: {r}")
            return None
        if r is None:
            logger.warning(f"[{pair['title']}] empty vote response")
            return None
        verdicts.append({"reasoning": r.reasoning, "correct": r.correct})
        if r.correct:
            correct_votes += 1

    incorrect_votes = VOTE_COUNT - correct_votes
    is_correct = correct_votes == VOTE_COUNT
    elapsed = time.monotonic() - t0

    result = {
        **pair,
        "correct": is_correct,
        "votes_correct": correct_votes,
        "votes_incorrect": incorrect_votes,
        "verdicts": verdicts,
        "verify_time_s": round(elapsed, 1),
    }

    status = "CORRECT" if is_correct else "WRONG"
    logger.info(f"[{pair['title']}] {status} ({correct_votes}/{VOTE_COUNT}) in {elapsed:.1f}s")
    return result


async def main(input_path: Path, output_path: Path) -> None:
    pairs = load_json_or_jsonl(input_path)
    logger.info(f"Loaded {len(pairs)} QA pairs from {input_path}")

    article_ids = {p["article_id"] for p in pairs}
    logger.info(f"Fetching {len(article_ids)} unique articles...")
    article_texts = await fetch_articles_by_id(article_ids)
    logger.info(f"Fetched {len(article_texts)} article texts")

    client = make_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    output_path.write_text("")

    results: list[dict] = []
    t0 = time.monotonic()
    batch_size = MAX_CONCURRENT

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        coros = []
        for p in batch:
            text = article_texts.get(p["article_id"], "")
            if not text:
                logger.warning(f"[{p['title']}] article_id={p['article_id']} not found, skipping")
                continue
            coros.append(verify_pair(client, semaphore, p, text))

        batch_results = await asyncio.gather(*coros, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, Exception):
                logger.error(f"Verify error: {r}")
                continue
            if r:
                results.append(r)

        output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
        logger.info(f"Progress: {len(results)}/{len(pairs)} verified ({time.monotonic()-t0:.1f}s)")

    elapsed = time.monotonic() - t0
    if results:
        correct_count = sum(1 for r in results if r["correct"])
        wrong_count = len(results) - correct_count
        split_count = sum(1 for r in results if not r["correct"] and r["votes_correct"] > 0)
        logger.info(f"=== Summary ({len(results)} pairs, {elapsed:.1f}s) ===")
        logger.info(f"Correct: {correct_count} ({correct_count/len(results)*100:.1f}%)")
        logger.info(f"Wrong:   {wrong_count} (unanimous: {wrong_count - split_count}, split: {split_count})")
        logger.info(f"Output:  {output_path}")

        # Per-difficulty breakdown
        by_diff: dict[str, list[dict]] = {}
        for r in results:
            d = r.get("difficulty", "unknown")
            by_diff.setdefault(d, []).append(r)
        for diff, items in sorted(by_diff.items()):
            c = sum(1 for r in items if r["correct"])
            logger.info(f"  {diff:6s}: {c}/{len(items)} correct ({c/len(items)*100:.1f}%)")
    else:
        logger.warning("No results to summarize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify golden QA pairs with 4-vote LLM judge")
    parser.add_argument("input", nargs="?", default="qa_pairs.jsonl", help="Input JSON/JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file (default: <input_stem>_verified.jsonl)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_verified").with_suffix(".jsonl")
    asyncio.run(main(in_path, out_path))

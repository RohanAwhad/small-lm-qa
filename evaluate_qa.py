"""Evaluate QA pair quality using DeepSeek as an LLM judge."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# --- Config ---
DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_CONCURRENT = 50
INPUT_FILE = Path("qa_pairs.jsonl")
OUTPUT_FILE = Path("qa_pairs_evaluated.jsonl")
LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "evaluate_qa.log", level=log_level, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}")

# --- Pydantic models ---
class EvalScore(BaseModel):
    faithfulness: int = Field(description="Score 1-5: how factually grounded the answer is in the Wikipedia text")
    completeness: int = Field(description="Score 1-5: how completely the answer addresses the question")
    clarity: int = Field(description="Score 1-5: how clear and well-written the answer is")
    reasoning: str = Field(description="Brief explanation of the scores")

class EvalResponse(BaseModel):
    evaluation: EvalScore

# --- Prompts ---
JUDGE_SYSTEM_PROMPT = """You are an expert QA evaluator. You will be given a Wikipedia article, a question, and an answer generated from that article.

Rate the answer on three criteria (1-5 scale):

1. **faithfulness**: Is the answer factually grounded in the Wikipedia text? 1 = major hallucination, 5 = perfectly accurate
2. **completeness**: Does the answer fully address the question? 1 = misses key points, 5 = thorough and complete
3. **clarity**: Is the answer well-written and understandable? 1 = confusing, 5 = clear and well-structured

Be strict but fair. A perfect score (5) should be rare — only for answers that are entirely accurate, complete, and clear.

Respond in JSON format: {"faithfulness": int, "completeness": int, "clarity": int, "reasoning": "..."}"""

JUDGE_USER_TEMPLATE = """## Wikipedia Article
{text}

## Question
{question}

## Generated Answer
{answer}

Evaluate this answer."""


# --- HF API for fetching article text ---
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"
HF_MAX_PER_REQUEST = 100


async def fetch_articles_by_id(article_ids: set[int]) -> dict[int, str]:
    """Fetch article texts by their row indices from HF API."""
    texts: dict[int, str] = {}
    async with httpx.AsyncClient(timeout=30) as http:
        for offset in range(0, max(article_ids) + 1, HF_MAX_PER_REQUEST):
            length = HF_MAX_PER_REQUEST
            ids_in_range = {i for i in article_ids if offset <= i < offset + length}
            if not ids_in_range:
                continue
            logger.debug(f"HF API request: offset={offset}, length={length}")
            resp = await http.get(
                HF_ROWS_URL,
                params={"dataset": HF_DATASET, "config": HF_CONFIG, "split": "train", "offset": offset, "length": length},
            )
            resp.raise_for_status()
            rows = resp.json()["rows"]
            for r in rows:
                idx = r["row_idx"]
                if idx in article_ids:
                    texts[idx] = r["row"]["text"]
            logger.info(f"Fetched {len(texts)}/{len(article_ids)} articles")
    return texts


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=BASE_URL,
    )


async def evaluate_pair(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pair: dict,
    article_text: str,
) -> dict | None:
    """Evaluate one QA pair."""
    async with semaphore:
        question = pair["question"]
        answer = pair["answer"]

        logger.debug(f"[{pair['title']}] evaluating Q: {question[:60]}...")
        t0 = time.monotonic()

        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(text=article_text, question=question, answer=answer)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            # thinking enabled by default
        )

        elapsed = time.monotonic() - t0
        raw = response.choices[0].message.content
        if not raw:
            logger.warning(f"[{pair['title']}] empty response")
            return None

        parsed = json.loads(raw)
        # Normalize: model might return eval dict directly or nested
        eval_data = parsed.get("evaluation", parsed)

        result = {
            **pair,
            "faithfulness": eval_data["faithfulness"],
            "completeness": eval_data["completeness"],
            "clarity": eval_data["clarity"],
            "judge_reasoning": eval_data.get("reasoning", ""),
            "eval_time_s": round(elapsed, 1),
        }

        logger.info(f"[{pair['title']}] eval: F={result['faithfulness']} C={result['completeness']} Cl={result['clarity']} in {elapsed:.1f}s")
        return result


async def main() -> None:
    # Load QA pairs
    pairs = [json.loads(l) for l in INPUT_FILE.read_text().strip().splitlines() if l.strip()]
    logger.info(f"Loaded {len(pairs)} QA pairs from {INPUT_FILE}")

    # Fetch article texts
    article_ids = {p["article_id"] for p in pairs}
    logger.info(f"Fetching {len(article_ids)} unique articles...")
    article_texts = await fetch_articles_by_id(article_ids)
    logger.info(f"Fetched {len(article_texts)} article texts")

    # Evaluate
    client = make_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    OUTPUT_FILE.write_text("")

    # Prepare task list, skipping articles not found
    eval_tasks: list[dict] = []
    for pair in pairs:
        text = article_texts.get(pair["article_id"], "")
        if not text:
            logger.warning(f"[{pair['title']}] article_id={pair['article_id']} not found, skipping")
            continue
        eval_tasks.append(pair)

    results: list[dict] = []
    t0 = time.monotonic()
    batch_size = 50
    for i in range(0, len(eval_tasks), batch_size):
        batch = eval_tasks[i : i + batch_size]
        coros = [evaluate_pair(client, semaphore, p, article_texts[p["article_id"]]) for p in batch]
        batch_results = await asyncio.gather(*coros, return_exceptions=True)

        for r in batch_results:
            if isinstance(r, Exception):
                logger.error(f"Eval error: {r}")
                continue
            if r:
                results.append(r)

        # Checkpoint write after each batch
        OUTPUT_FILE.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
        )
        logger.info(f"Progress: {len(results)}/{len(eval_tasks)} evaluated ({time.monotonic()-t0:.1f}s)")

    # Summary stats
    elapsed = time.monotonic() - t0
    if results:
        scores = [(r["faithfulness"], r["completeness"], r["clarity"]) for r in results]
        avg_f = sum(s[0] for s in scores) / len(scores)
        avg_c = sum(s[1] for s in scores) / len(scores)
        avg_cl = sum(s[2] for s in scores) / len(scores)

        logger.info(f"=== Summary ({len(results)} pairs, {elapsed:.1f}s) ===")
        logger.info(f"Faithfulness:  {avg_f:.2f} / 5")
        logger.info(f"Completeness:  {avg_c:.2f} / 5")
        logger.info(f"Clarity:       {avg_cl:.2f} / 5")
        logger.info(f"Overall:       {(avg_f+avg_c+avg_cl)/3:.2f} / 5")
    else:
        logger.warning("No results to summarize")


if __name__ == "__main__":
    asyncio.run(main())

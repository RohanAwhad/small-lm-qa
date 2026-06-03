"""Evaluate Gemma3 answers against DeepSeek golden answers."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI

DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_CONCURRENT = 50
INPUT_FILE = Path("qa_pairs_gemma.jsonl")
OUTPUT_FILE = Path("qa_pairs_gemma_eval.jsonl")
LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "compare_gemma.log", level="DEBUG", rotation="10 MB")

JUDGE_SYSTEM_PROMPT = """You are an expert QA evaluator. Given a Wikipedia article, a question, a golden answer (reference), and a model answer, evaluate the model answer.

Rate the model answer on four criteria (1-5 scale):

1. **faithfulness**: Is the model answer factually grounded in the Wikipedia article? 1 = major hallucination, 5 = perfectly accurate
2. **completeness**: Does the model answer fully address the question? 1 = misses key points, 5 = thorough
3. **clarity**: Is the model answer clear and well-written? 1 = confusing, 5 = well-structured
4. **correctness**: How well does the model answer match the golden answer in terms of factual accuracy and completeness? 1 = completely wrong/missing, 5 = matches golden answer perfectly"""

JUDGE_USER_TEMPLATE = """## Wikipedia Article
{text}

## Question
{question}

## Golden Answer (Reference)
{golden}

## Model Answer (to evaluate)
{answer}

Evaluate the model answer. Respond in JSON format: {{"faithfulness": int, "completeness": int, "clarity": int, "correctness": int, "reasoning": "..."}}"""

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


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)


async def evaluate_pair(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pair: dict,
    article_text: str,
) -> dict | None:
    async with semaphore:
        question = pair["question"]
        golden = pair["answer"]
        answer = pair["model_answer"]
        title = pair["title"]

        logger.debug(f"[{title}] evaluating gemma vs golden...")
        t0 = time.monotonic()

        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(text=article_text, question=question, golden=golden, answer=answer)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        elapsed = time.monotonic() - t0
        raw = response.choices[0].message.content
        if not raw:
            logger.warning(f"[{title}] empty response")
            return None

        parsed = json.loads(raw)
        eval_data = parsed.get("evaluation", parsed)

        result = {
            **pair,
            "faithfulness": eval_data["faithfulness"],
            "completeness": eval_data["completeness"],
            "clarity": eval_data["clarity"],
            "correctness": eval_data["correctness"],
            "judge_reasoning": eval_data.get("reasoning", ""),
            "eval_time_s": round(elapsed, 1),
        }

        logger.info(f"[{title}] F={result['faithfulness']} C={result['completeness']} OK={result['correctness']} in {elapsed:.1f}s")
        return result


async def main() -> None:
    pairs = [json.loads(l) for l in INPUT_FILE.read_text().strip().splitlines() if l.strip()]
    logger.info(f"Loaded {len(pairs)} pairs from {INPUT_FILE}")

    article_ids = {p["article_id"] for p in pairs}
    logger.info(f"Fetching {len(article_ids)} unique articles...")
    article_texts = await fetch_articles_by_id(article_ids)
    logger.info(f"Fetched {len(article_texts)} article texts")

    client = make_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    results: list[dict] = []
    t0 = time.monotonic()
    batch_size = 50
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        coros = []
        for p in batch:
            text = article_texts.get(p["article_id"], "")
            if not text:
                continue
            coros.append(evaluate_pair(client, semaphore, p, text))
        batch_results = await asyncio.gather(*coros, return_exceptions=True)

        for r in batch_results:
            if isinstance(r, Exception):
                logger.error(f"Error: {r}")
                continue
            if r:
                results.append(r)

        OUTPUT_FILE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
        logger.info(f"Progress: {len(results)}/{len(pairs)} evaluated ({time.monotonic()-t0:.1f}s)")

    elapsed = time.monotonic() - t0
    if results:
        fs = [r["faithfulness"] for r in results]
        cs = [r["completeness"] for r in results]
        cls = [r["clarity"] for r in results]
        ok = [r["correctness"] for r in results]
        logger.info(f"=== Summary ({len(results)} pairs, {elapsed:.1f}s) ===")
        logger.info(f"Faithfulness:  {sum(fs)/len(fs):.2f} / 5")
        logger.info(f"Completeness:  {sum(cs)/len(cs):.2f} / 5")
        logger.info(f"Clarity:       {sum(cls)/len(cls):.2f} / 5")
        logger.info(f"Correctness:   {sum(ok)/len(ok):.2f} / 5")
        logger.info(f"Overall:       {(sum(fs)+sum(cs)+sum(cls)+sum(ok))/(len(fs)*4):.2f} / 5")


if __name__ == "__main__":
    asyncio.run(main())

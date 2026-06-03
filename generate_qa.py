"""Generate QA pairs from Wikipedia articles using DeepSeek V4 Pro."""

import asyncio
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

# --- Config ---
DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_CONCURRENT = 10
OUTPUT_FILE = Path("qa_pairs.jsonl")
LOG_DIR = Path("logs")

# --- Logging setup ---
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "generate_qa.log", level=log_level, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}", rotation="10 MB")


# --- Pydantic models for structured output ---
class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"
    unknown = "unknown"


class QAPair(BaseModel):
    difficulty: Difficulty
    question: str
    answer: str


class QAResponse(BaseModel):
    qa_pairs: list[QAPair]


# --- Prompts ---
SYSTEM_PROMPT = """You are an expert question-answer pair generator. Given a Wikipedia article, generate exactly 15 question-answer pairs based ONLY on the provided text.

Generate:
- 5 EASY questions: answers should be 1-2 sentences
- 5 MEDIUM questions: answers should be 3-5 sentences
- 5 HARD questions: answers should be 6-10 sentences

Rules:
- Questions must be answerable from the provided text alone
- Answers must be factually grounded in the text
- Easy questions test simple recall of facts
- Medium questions require connecting multiple facts
- Hard questions require synthesis, comparison, or multi-step reasoning
- Vary question types (who, what, when, why, how, compare, explain)
- Answers must be thorough: cover all relevant subtopics and supporting details present in the text, not just the main point

Respond with this exact JSON structure:
{
  "qa_pairs": [
    {"difficulty": "easy", "question": "...", "answer": "..."},
    {"difficulty": "medium", "question": "...", "answer": "..."},
    {"difficulty": "hard", "question": "...", "answer": "..."}
  ]
}
Where difficulty is one of "easy", "medium", "hard", and there are exactly 5 of each difficulty (15 total)."""

USER_PROMPT_TEMPLATE = """Article Title: {title}

Article Text:
{text}

Generate 15 question-answer pairs (5 easy, 5 medium, 5 hard) based on this article. Respond in JSON."""


# --- HF dataset fetching ---
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"
HF_MAX_PER_REQUEST = 100


async def fetch_wikipedia_articles(n: int) -> list[dict]:
    """Fetch n articles from Wikipedia via HF datasets server API."""
    articles: list[dict] = []
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=30) as http:
        for offset in range(0, n, HF_MAX_PER_REQUEST):
            length = min(HF_MAX_PER_REQUEST, n - offset)
            logger.debug(f"HF API request: offset={offset}, length={length}")
            resp = await http.get(
                HF_ROWS_URL,
                params={"dataset": HF_DATASET, "config": HF_CONFIG, "split": "train", "offset": offset, "length": length},
            )
            resp.raise_for_status()
            rows = resp.json()["rows"]
            for r in rows:
                row = r["row"]
                articles.append({"article_id": r["row_idx"], "title": row["title"], "text": row["text"]})
            logger.info(f"Fetched {len(articles)}/{n} articles ({time.monotonic()-t0:.1f}s)")
    return articles


# --- DeepSeek QA generation ---
def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=BASE_URL,
    )


async def generate_qa_pairs(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    article_id: int,
    title: str,
    text: str,
) -> list[dict]:
    """Call DeepSeek to generate QA pairs for one article."""
    async with semaphore:
        logger.debug(f"[{title}] starting API call (text={len(text)} chars)")
        t0 = time.monotonic()

        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(title=title, text=text)},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            # thinking enabled by default
        )

        elapsed = time.monotonic() - t0
        choice = response.choices[0]
        logger.debug(f"[{title}] API response in {elapsed:.1f}s, finish_reason={choice.finish_reason}, usage={response.usage}")

        raw = choice.message.content
        if not raw:
            logger.warning(f"[{title}] empty response content")
            return []

        logger.debug(f"[{title}] parsing JSON ({len(raw)} chars): {raw[:300]}...")
        parsed = json.loads(raw)

        # Normalize to {qa_pairs: [...]} with difficulty on each item
        if isinstance(parsed, list):
            # Flat list — assume first 5 easy, next 5 medium, last 5 hard
            difficulties = ["easy"] * 5 + ["medium"] * 5 + ["hard"] * 5
            for i, item in enumerate(parsed):
                item["difficulty"] = difficulties[i] if i < len(difficulties) else "unknown"
            parsed = {"qa_pairs": parsed}
            logger.debug(f"[{title}] flat list -> qa_pairs ({len(parsed['qa_pairs'])} pairs)")
        elif "qa_pairs" not in parsed:
            # Grouped by difficulty keys: {"easy": [...], "medium": [...], "hard": [...]}
            normalized = []
            for difficulty in ("easy", "medium", "hard"):
                for item in parsed.get(difficulty, []):
                    item["difficulty"] = difficulty
                    normalized.append(item)
            parsed = {"qa_pairs": normalized}
            logger.debug(f"[{title}] grouped -> qa_pairs ({len(normalized)} pairs)")
        else:
            # Already has qa_pairs key — ensure difficulty is set
            for item in parsed["qa_pairs"]:
                item.setdefault("difficulty", "unknown")

        qa_response = QAResponse.model_validate(parsed)
        pairs = [
            {
                "difficulty": p.difficulty.value,
                "question": p.question,
                "answer": p.answer,
                "title": title,
                "source_text_length": len(text),
                "article_id": article_id,
            }
            for p in qa_response.qa_pairs
        ]

        logger.info(f"[{title}] -> {len(pairs)} pairs in {elapsed:.1f}s")
        return pairs


async def process_batch(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    articles: list[dict],
) -> list[dict]:
    """Process a batch of articles concurrently."""
    tasks = [
        generate_qa_pairs(client, semaphore, art["article_id"], art["title"], art["text"])
        for art in articles
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_pairs: list[dict] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"[{articles[i]['title']}] {type(result).__name__}: {result}")
            continue
        all_pairs.extend(result)
    return all_pairs


def write_jsonl(pairs: list[dict], path: Path, mode: str = "a") -> None:
    with open(path, mode) as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


async def main() -> None:
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    logger.info(f"Starting: n_topics={n_topics}, model={DEEPSEEK_MODEL}, concurrency={MAX_CONCURRENT}")

    articles = await fetch_wikipedia_articles(n_topics)
    logger.info(f"Loaded {len(articles)} articles")

    client = make_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    OUTPUT_FILE.write_text("")

    batch_size = 50
    total_pairs = 0
    t0 = time.monotonic()
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        logger.info(f"Batch {i // batch_size + 1} ({len(batch)} articles)...")
        pairs = await process_batch(client, semaphore, batch)
        write_jsonl(pairs, OUTPUT_FILE)
        total_pairs += len(pairs)
        logger.info(f"Checkpoint: {total_pairs} pairs written ({time.monotonic()-t0:.1f}s elapsed)")

    logger.info(f"Done. {total_pairs} QA pairs -> {OUTPUT_FILE} in {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

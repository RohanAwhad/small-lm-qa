"""Judge DPO rollouts: pick best/worst response per question via LLM-as-judge.

Takes a JSONL file where each line has:
  - question, golden_answer, context, rollouts (list of model responses)

Sends all rollouts per question to DeepSeek Flash, which picks best/worst idx.
Judge evaluates reasoning quality, answer correctness, hallucination, and coherence.

Output: JSONL with judgment attached (best_idx, worst_idx, all_bad, all_good, explanation).
Resume support: appends results, skips already-judged questions on restart.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "20"))
LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "judge_dpo_rollouts.log", level=log_level, rotation="10 MB")


class JudgeResult(BaseModel):
    best_idx: int | None
    worst_idx: int | None
    all_bad: bool
    all_good: bool
    explanation: str


JUDGE_SYSTEM = """You are a judge comparing multiple student responses to the same question.
You will see the question, the source context, the reference answer, and all student responses.

Each student response may include <reasoning>...</reasoning> tags showing its thought process, followed by the final answer.

For each response, check:
- Does the reasoning hallucinate facts not in the context?
- Does the reasoning logically lead to the final answer?
- Does the reasoning make sense given the context?
- Is the final answer factually correct compared to the reference?
- Is the response complete or does it cut off mid-sentence?

Pick the BEST response index and the WORST response index.
If all responses are equally bad (none is correct), set "all_bad": true and best_idx/worst_idx to null.
If all responses are equally good, set "all_good": true and best_idx/worst_idx to null.

Respond only in json format."""

JUDGE_USER = """Context:
{context}

Question: {question}

Reference answer: {reference}

Student responses:
{responses}

Respond in JSON: {{"best_idx": <0-based index or null>, "worst_idx": <0-based index or null>, "all_bad": <true/false>, "all_good": <true/false>, "explanation": "brief justification for picks"}}"""


def format_responses(rollouts: list[str]) -> str:
    """Format rollouts for the judge prompt."""
    parts = []
    for i, r in enumerate(rollouts):
        parts.append(f"[Response {i}]\n{r}")
    return "\n\n".join(parts)


def make_key(article_id: int, question: str) -> str:
    return f"{article_id}::{question}"


def load_done_keys(output_path: Path) -> set[str]:
    """Load already-judged keys for resume support."""
    if not output_path.exists():
        return set()
    keys = set()
    for line in output_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "judgment" in r:
            keys.add(make_key(r["article_id"], r["question"]))
    logger.info(f"Resume: {len(keys)} questions already judged in {output_path}")
    return keys


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=10, jitter=2),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError, Exception)),
    before_sleep=lambda rs: logger.warning(
        f"Judge retry {rs.attempt_number}/5: {rs.outcome.exception()!r}"
    ),
    reraise=True,
)
async def call_judge(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    r: dict,
) -> JudgeResult:
    """Judge a single question's rollouts with retry. Semaphore outside retry."""
    async with semaphore:
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(
                    context=r["context"],
                    question=r["question"],
                    reference=r["golden_answer"],
                    responses=format_responses(r["rollouts"]),
                )},
            ],
            response_format={"type": "json_object"},
        )
    content = resp.choices[0].message.content
    assert content, "Empty LLM response"
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return JudgeResult.model_validate_json(content)


async def main(input_path: Path, output_path: Path) -> None:
    # Load all records from input
    all_records = [json.loads(l) for l in input_path.read_text().strip().splitlines() if l.strip()]
    logger.info(f"Loaded {len(all_records)} records from {input_path}")

    # Resume support
    done_keys = load_done_keys(output_path)
    todo = [r for r in all_records if make_key(r["article_id"], r["question"]) not in done_keys]
    logger.info(f"TODO: {len(todo)} questions ({len(done_keys)} already judged)")

    if not todo:
        logger.info("Nothing to do — all questions already judged")
        return

    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Append mode for crash-safe checkpointing
    f = open(output_path, "a")
    written = 0
    failed = 0
    lock = asyncio.Lock()

    async def judge_and_write(r: dict) -> None:
        nonlocal written, failed
        judgment = await call_judge(client, semaphore, r)
        r["judgment"] = judgment.model_dump()

        async with lock:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written % 100 == 0:
                logger.info(f"Progress: {written}/{len(todo)} judged, {failed} failed")

    # Process all
    t0 = time.monotonic()
    results = await asyncio.gather(
        *[judge_and_write(r) for r in todo], return_exceptions=True,
    )
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed += 1
            logger.error(f"Failed article_id={todo[i].get('article_id')} "
                         f"q='{todo[i]['question'][:60]}': {result}")

    f.close()
    elapsed = time.monotonic() - t0

    n_pairs = sum(1 for r in todo if "judgment" in r
                  and not r["judgment"].get("all_bad")
                  and not r["judgment"].get("all_good")
                  and r["judgment"].get("best_idx") is not None
                  and r["judgment"].get("worst_idx") is not None)

    logger.info(
        f"Done. {written} judged, {failed} failed, {n_pairs} usable pairs "
        f"in {elapsed:.1f}s ({written / max(elapsed, 0.1):.1f} q/s). Output: {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge DPO rollouts via LLM-as-judge")
    parser.add_argument("input", help="Input JSONL with rollouts per question")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL with judgments")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_judged")
    asyncio.run(main(in_path, out_path))

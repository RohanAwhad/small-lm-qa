"""Judge DPO rollouts: pick best/worst response per question via LLM-as-judge.

Takes a JSONL file where each line has:
  - question, golden_answer, context, rollouts (list of model responses)

Sends all rollouts per question to DeepSeek Flash, which picks best/worst idx.
Judge evaluates reasoning quality, answer correctness, hallucination, and coherence.

Output: JSONL with judgment attached (best_idx, worst_idx, all_bad, all_good, explanation).
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

DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_CONCURRENT = 50
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


def strip_reasoning(text: str) -> str:
    return re.sub(r"<reasoning>.*?</reasoning?>", "", text, flags=re.DOTALL).strip()


async def judge_all(records: list[dict]) -> list[dict]:
    """Send all rollouts per question to DeepSeek Flash, ask for best/worst."""
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=BASE_URL,
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def judge_one(r: dict) -> JudgeResult:
        for attempt in range(3):
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
                content = resp.choices[0].message.content or ""
                content = content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                try:
                    return JudgeResult.model_validate_json(content)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"[retry {attempt+1}/3] Bad response for article_id={r.get('article_id')}: {e}")
        return JudgeResult(best_idx=None, worst_idx=None, all_bad=True, all_good=False, explanation="judge failed")

    logger.info(f"Judging {len(records)} questions with {DEEPSEEK_MODEL}...")
    t0 = time.monotonic()
    judgments = await asyncio.gather(*[judge_one(r) for r in records])
    elapsed = time.monotonic() - t0
    logger.info(f"Judging done: {elapsed:.1f}s ({len(records) / elapsed:.1f} q/s)")

    for r, j in zip(records, judgments):
        r["judgment"] = j.model_dump()

    return records


async def main(input_path: Path, output_path: Path) -> None:
    # Load rollouts
    records = [json.loads(l) for l in input_path.read_text().strip().splitlines() if l.strip()]
    logger.info(f"Loaded {len(records)} records from {input_path}")

    # Judge
    records = await judge_all(records)

    # Write output
    n_pairs = 0
    n_skipped = 0
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            j = r["judgment"]
            if j["all_bad"] or j["all_good"] or j["best_idx"] is None or j["worst_idx"] is None:
                n_skipped += 1
            else:
                n_pairs += 1

    logger.info(f"Done. {n_pairs} usable DPO pairs, {n_skipped} skipped. Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge DPO rollouts via LLM-as-judge")
    parser.add_argument("input", help="Input JSONL with rollouts per question")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL with judgments")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_judged")
    asyncio.run(main(in_path, out_path))

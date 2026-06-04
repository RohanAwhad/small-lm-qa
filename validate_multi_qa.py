"""Validate multi-article QA pairs using RAGAS-style metrics.

Two metrics (from RAGAS paper, arxiv 2309.15217):

  1. Faithfulness: Are claims in reference_answer inferable from source articles?
     - Decompose answer -> atomic statements (LLM call 1)
     - Classify each statement against source article text (LLM call 2)
     - Score = |supported| / |total statements|

  2. Context Relevance: Is the exploration context focused on the question?
     - Given question + exploration_log, extract relevant sentences (LLM call 3)
     - Score = |relevant sentences| / |total sentences in context|

Usage:
  uv run python validate_multi_qa.py [input.jsonl] [-o output.jsonl]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from utils.wikipedia_loader import load_articles_by_id

DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_CONCURRENT = 10
MAX_ARTICLE_CHARS = 40_000
LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(
    LOG_DIR / "validate_multi_qa.log",
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
)


# --- Pydantic models ---


class DecompOutput(BaseModel):
    reasoning: str = ""
    statements: list[str]


class FaithfulnessOutput(BaseModel):
    reasoning: str = ""
    verdicts: dict[str, str]


class ContextRelevanceOutput(BaseModel):
    reasoning: str = ""
    relevant_indices: list[int]


# --- Prompts ---

DECOMPOSE_SYSTEM = "You decompose answers into atomic factual statements. Respond only in json format."

DECOMPOSE_USER = """Break this reference answer into atomic factual statements. Each statement must be:
1. A single, independently verifiable claim
2. Self-contained (understandable without reading the full answer)
3. Factual (not opinions, hedges, or structural phrases like "in conclusion")

Reference Answer:
{answer}

Respond in JSON: {{"reasoning": "brief note", "statements": ["statement1", "statement2", ...]}}"""


FAITHFULNESS_SYSTEM = """You verify factual statements against source text.
For each statement, classify as:
- SUPPORTED: the statement can be inferred from the source text
- NOT_SUPPORTED: the statement cannot be inferred from or contradicts the source text

Be strict: the source text must contain sufficient evidence to support the claim.
Respond only in json format."""

FAITHFULNESS_USER = """Source text (from Wikipedia articles):
{context}

Statements to verify:
{statements}

For each statement, classify as SUPPORTED or NOT_SUPPORTED.

Respond in JSON: {{"reasoning": "...", "verdicts": {{"statement text": "SUPPORTED"|"NOT_SUPPORTED"}}}}"""


CONTEXT_REL_SYSTEM = """You assess whether retrieved context sentences are relevant to a question.
Given a question and numbered context sentences, identify which sentence indices
contain information needed to answer the question.
Respond only in json format."""

CONTEXT_REL_USER = """Question:
{question}

Numbered context sentences:
{numbered_sentences}

Which sentence indices contain information needed to answer the question?
Return ONLY the indices of relevant sentences.

Respond in JSON: {{"reasoning": "...", "relevant_indices": [0, 3, 7, ...]}}"""


# --- LLM ---


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=3, jitter=1),
    retry=retry_if_exception_type((json.JSONDecodeError, AssertionError)),
    before_sleep=lambda rs: logger.warning(
        f"LLM call retry {rs.attempt_number}/5: {rs.outcome.exception()!r}"
    ),
    reraise=True,
)
async def call_llm_json(client: AsyncOpenAI, system: str, user: str) -> dict[str, Any]:
    t0 = time.monotonic()
    resp = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    assert content, "Empty LLM response"
    elapsed = time.monotonic() - t0
    logger.debug(f"LLM json call took {elapsed:.1f}s, {len(content)} chars")
    return json.loads(content)


# --- Faithfulness ---


async def compute_faithfulness(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    reference_answer: str,
    article_texts: dict[int, str],
) -> dict[str, Any]:
    """Faithfulness = |supported statements| / |total statements|."""
    async with semaphore:
        t0 = time.monotonic()

        # Step 1: decompose answer into atomic statements
        raw1 = await call_llm_json(
            client, DECOMPOSE_SYSTEM,
            DECOMPOSE_USER.format(answer=reference_answer),
        )
        decomp = DecompOutput.model_validate(raw1)
        statements = decomp.statements

        if not statements:
            logger.warning("No statements extracted from reference answer")
            return {
                "faithfulness": 0.0, "statements": [],
                "verdicts": {}, "n_supported": 0, "n_total": 0,
                "eval_time_s": round(time.monotonic() - t0, 1),
            }

        # Step 2: build context from source articles (truncated)
        context_parts = []
        for aid in sorted(article_texts):
            text = article_texts[aid][:MAX_ARTICLE_CHARS]
            context_parts.append(f"=== Article {aid} ===\n{text}")
        context = "\n\n".join(context_parts)

        # Step 3: classify each statement against context
        stmt_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
        raw2 = await call_llm_json(
            client, FAITHFULNESS_SYSTEM,
            FAITHFULNESS_USER.format(context=context, statements=stmt_text),
        )
        faith = FaithfulnessOutput.model_validate(raw2)

        n_supported = sum(1 for v in faith.verdicts.values() if v == "SUPPORTED")
        n_total = len(statements)
        score = n_supported / n_total if n_total > 0 else 0.0
        elapsed = time.monotonic() - t0

        logger.info(
            f"Faithfulness: {score:.3f} ({n_supported}/{n_total} supported) "
            f"in {elapsed:.1f}s"
        )
        return {
            "faithfulness": round(score, 4),
            "statements": statements,
            "verdicts": faith.verdicts,
            "n_supported": n_supported,
            "n_total": n_total,
            "eval_time_s": round(elapsed, 1),
        }


# --- Context Relevance ---

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out very short fragments."""
    raw = _SENT_SPLIT.split(text)
    return [s.strip() for s in raw if len(s.strip()) > 15]


async def compute_context_relevance(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    question: str,
    exploration_log: str,
) -> dict[str, Any]:
    """Context relevance = |relevant sentences| / |total sentences in context|."""
    async with semaphore:
        t0 = time.monotonic()

        sentences = split_sentences(exploration_log)
        n_total = len(sentences)
        if n_total == 0:
            return {
                "context_relevance": 0.0,
                "n_relevant_sentences": 0, "n_total_sentences": 0,
                "relevant_indices": [], "eval_time_s": 0.0,
            }

        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
        raw = await call_llm_json(
            client, CONTEXT_REL_SYSTEM,
            CONTEXT_REL_USER.format(question=question, numbered_sentences=numbered),
        )
        cr = ContextRelevanceOutput.model_validate(raw)

        # Clamp indices to valid range
        valid_indices = [idx for idx in cr.relevant_indices if 0 <= idx < n_total]
        n_relevant = len(valid_indices)
        score = n_relevant / n_total
        elapsed = time.monotonic() - t0

        logger.info(
            f"Context relevance: {score:.3f} ({n_relevant}/{n_total} relevant) "
            f"in {elapsed:.1f}s"
        )
        return {
            "context_relevance": round(score, 4),
            "n_relevant_sentences": n_relevant,
            "n_total_sentences": n_total,
            "relevant_indices": valid_indices,
            "eval_time_s": round(elapsed, 1),
        }


# --- Main ---


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for line in path.read_text().strip().splitlines():
        if line.strip():
            results.append(json.loads(line))
    return results


def qa_key(pair: dict[str, Any]) -> str:
    return pair["question"][:200]


def extract_article_ids(pair: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for a in pair.get("articles_used", []):
        aid = a["article_id"] if isinstance(a, dict) else a
        ids.append(aid)
    return ids


async def main(input_path: Path, output_path: Path) -> None:
    pairs = load_jsonl(input_path)
    logger.info(f"Loaded {len(pairs)} multi-article QA pairs from {input_path}")

    client = make_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Collect all article IDs needed
    all_article_ids: set[int] = set()
    for p in pairs:
        all_article_ids.update(extract_article_ids(p))

    logger.info(f"Loading {len(all_article_ids)} source articles for faithfulness")
    article_texts = load_articles_by_id(all_article_ids)
    logger.info(f"Loaded {len(article_texts)} articles")

    # Resume support
    completed_keys: set[str] = set()
    results: list[dict[str, Any]] = []
    if output_path.exists():
        for line in output_path.read_text().strip().splitlines():
            if line.strip():
                r = json.loads(line)
                completed_keys.add(qa_key(r))
                results.append(r)
        logger.info(f"Resuming: {len(results)} already evaluated")

    pending = [p for p in pairs if qa_key(p) not in completed_keys]
    if pending:
        logger.info(f"Evaluating {len(pending)} pending pairs")
    else:
        logger.info("All pairs already evaluated")

    t0 = time.monotonic()
    for i, pair in enumerate(pending):
        question = pair["question"]
        reference_answer = pair["reference_answer"]
        exploration_log = pair.get("exploration_log", "")

        pair_aids = extract_article_ids(pair)
        pair_texts = {aid: article_texts[aid] for aid in pair_aids if aid in article_texts}

        logger.info(f"[{i + 1}/{len(pending)}] {question[:80]}...")

        # Run both metrics concurrently
        faith_coro = compute_faithfulness(client, semaphore, reference_answer, pair_texts)
        if exploration_log:
            cr_coro = compute_context_relevance(client, semaphore, question, exploration_log)
            faith_result, cr_result = await asyncio.gather(faith_coro, cr_coro)
        else:
            faith_result = await faith_coro
            cr_result: dict[str, Any] = {
                "context_relevance": None,
                "n_relevant_sentences": 0, "n_total_sentences": 0,
                "relevant_sentences": [], "eval_time_s": 0,
            }

        result: dict[str, Any] = {
            "question": question,
            "articles_used": pair.get("articles_used", []),
            "reference_answer_chars": len(reference_answer),
            "exploration_log_chars": len(exploration_log),
            **{f"faith_{k}": v for k, v in faith_result.items()},
            **{f"ctx_{k}": v for k, v in cr_result.items()},
        }
        results.append(result)

        # Write incrementally
        output_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
        )
        logger.info(f"Progress: {len(results)}/{len(pairs)} evaluated")

    # Summary
    elapsed = time.monotonic() - t0
    if results:
        avg_faith = sum(r["faith_faithfulness"] for r in results) / len(results)
        ctx_scores = [r["ctx_context_relevance"] for r in results if r["ctx_context_relevance"] is not None]
        avg_ctx = sum(ctx_scores) / len(ctx_scores) if ctx_scores else 0.0

        logger.info(f"=== Summary ({len(results)} pairs, {elapsed:.1f}s) ===")
        logger.info(f"Avg Faithfulness:      {avg_faith:.3f}")
        logger.info(f"Avg Context Relevance: {avg_ctx:.3f}")
        logger.info(f"Output: {output_path}")
    else:
        logger.warning("No results to summarize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate multi-article QA with RAGAS-style metrics"
    )
    parser.add_argument(
        "input", nargs="?", default="deepresearch_qa_multi.jsonl",
        help="Input JSONL (default: deepresearch_qa_multi.jsonl)",
    )
    parser.add_argument("-o", "--output", default=None, help="Output JSONL")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_stem(in_path.stem + "_validated").with_suffix(".jsonl")
    )
    asyncio.run(main(in_path, out_path))

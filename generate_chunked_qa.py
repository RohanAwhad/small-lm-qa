"""Generate chunked QA pairs: recursive chunking + BM25 retrieval + answer regeneration.

Pipeline per article:
  1. Recursive-split article text into <=512 token chunks (Gemma3 tokenizer)
  2. For each QA pair, BM25-retrieve top-3 chunks using golden answer
  3. Regenerate answer from retrieved chunks (with reasoning_content)
  4. Evaluate context precision + recall (RAGAS-style)
  5. Write results to JSONL (one line per QA pair)

Checkpointing: resumes by skipping article_ids already in output file.

Usage:
  uv run python generate_chunked_qa.py [N]       # process N articles (default: all)
  uv run python generate_chunked_qa.py [N] [-o output.jsonl]
"""

import argparse
import asyncio
from asyncio import timeout as async_timeout
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger
from openai import APIError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from transformers import AutoTokenizer

from utils.wikipedia_loader import load_articles_by_id

# --- Config ---

TOKENIZER_ID = "unsloth/gemma-3-270m-it"
MAX_CHUNK_TOKENS = 512
TOP_K = 3
DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_CONCURRENT = 50
BATCH_SIZE = 20
INPUT_FILE = Path("qa_pairs.jsonl")
OUTPUT_FILE = Path("qa_pairs_chunked.jsonl")

SEPARATORS = ["\n\n", "\n", ". ", ", ", " "]

# --- Logging ---

LOG_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)
Path("logs").mkdir(exist_ok=True)
logger.add("logs/generate_chunked_qa.log", level="DEBUG", rotation="50 MB")


# --- Recursive chunking ---


def recursive_chunk(text: str, tokenizer, max_tokens: int, separators: list[str]) -> list[str]:
    """Recursively split text using separator hierarchy until all chunks <= max_tokens."""
    tok_len = len(tokenizer.encode(text, add_special_tokens=False))
    if tok_len <= max_tokens:
        return [text.strip()] if text.strip() else []

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    chunks.append(part + sep)
                else:
                    chunks.append(part)

            remaining_seps = separators[separators.index(sep) + 1:]
            result = []
            for chunk in chunks:
                if not chunk.strip():
                    continue
                c_len = len(tokenizer.encode(chunk, add_special_tokens=False))
                if c_len <= max_tokens:
                    result.append(chunk.strip())
                elif remaining_seps:
                    result.extend(recursive_chunk(chunk, tokenizer, max_tokens, remaining_seps))
                else:
                    ids = tokenizer.encode(chunk, add_special_tokens=False)[:max_tokens]
                    result.append(tokenizer.decode(ids).strip())
            return result

    ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return [tokenizer.decode(ids).strip()]


# --- BM25 retrieval ---


def retrieve_bm25(query: str, chunks: list[str], top_k: int = TOP_K) -> list[tuple[int, str, float]]:
    """Retrieve top_k chunks by BM25 score. Returns [(index, chunk_text, score)]."""
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(idx, chunks[idx], score) for idx, score in ranked]


# --- LLM helpers ---


class ChunkRelevanceOutput(BaseModel):
    reasoning: str = ""
    relevant_indices: list[int]


class DecompOutput(BaseModel):
    reasoning: str = ""
    statements: list[str]


class RecallVerdicts(BaseModel):
    reasoning: str = ""
    verdicts: dict[str, str]


CHUNK_REL_SYSTEM = """You assess whether retrieved text chunks are relevant to answering a question.
Given a question and numbered chunks, identify which chunk indices contain information
needed to answer the question. A chunk is relevant if it contains ANY useful information
for answering the question.
Respond only in json format."""

CHUNK_REL_USER = """Question:
{question}

Retrieved chunks:
{numbered_chunks}

Which chunk indices are relevant to answering the question?

Respond in JSON: {{"reasoning": "...", "relevant_indices": [0, 2, ...]}}"""

DECOMPOSE_SYSTEM = "You decompose answers into atomic factual statements. Respond only in json format."

DECOMPOSE_USER = """Break this reference answer into atomic factual statements. Each statement must be:
1. A single, independently verifiable claim
2. Self-contained (understandable without reading the full answer)
3. Factual (not opinions, hedges, or structural phrases like "in conclusion")

Reference Answer:
{answer}

Respond in JSON: {{"reasoning": "brief note", "statements": ["statement1", "statement2", ...]}}"""

RECALL_SYSTEM = """You verify factual statements against retrieved context chunks.
For each statement, classify as:
- SUPPORTED: the statement can be inferred from the context chunks
- NOT_SUPPORTED: the statement cannot be inferred from or contradicts the context

Be strict: the context must contain sufficient evidence to support the claim.
Respond only in json format."""

RECALL_USER = """Retrieved context:
{context}

Statements to verify:
{statements}

For each statement, classify as SUPPORTED or NOT_SUPPORTED.

Respond in JSON: {{"reasoning": "...", "verdicts": {{"statement text": "SUPPORTED"|"NOT_SUPPORTED"}}}}"""

REGEN_SYSTEM = """You answer questions using ONLY the provided context chunks.
Be thorough and detailed, but do NOT include any information not found in the context.
If the context does not contain enough information to fully answer the question, answer
with what is available and note what is missing."""

REGEN_USER = """Context:
{context}

Question:
{question}

Answer the question using ONLY the information in the context above."""


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=3, jitter=1),
    retry=retry_if_exception_type((json.JSONDecodeError, AssertionError, APIError, APITimeoutError, TimeoutError)),
    before_sleep=lambda rs: logger.warning(
        f"LLM json retry {rs.attempt_number}/5: {rs.outcome.exception()!r}"
    ),
    reraise=True,
)
async def call_llm_json(client: AsyncOpenAI, system: str, user: str) -> dict[str, Any]:
    async with async_timeout(60):
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
    return json.loads(content)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=3, jitter=1),
    retry=retry_if_exception_type((APIError, APITimeoutError, TimeoutError)),
    before_sleep=lambda rs: logger.warning(
        f"Regen retry {rs.attempt_number}/5: {rs.outcome.exception()!r}"
    ),
    reraise=True,
)
async def regenerate_answer(client: AsyncOpenAI, question: str, chunks: list[str]) -> tuple[str, str]:
    """Returns (answer, reasoning_content)."""
    context = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks))
    async with async_timeout(90):
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": REGEN_SYSTEM},
                {"role": "user", "content": REGEN_USER.format(context=context, question=question)},
            ],
        )
    msg = resp.choices[0].message
    answer = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    return answer, reasoning


# --- Evaluation ---


async def compute_context_precision(
    client: AsyncOpenAI, question: str, chunks: list[str]
) -> dict[str, Any]:
    n_total = len(chunks)
    if n_total == 0:
        return {"context_precision": 0.0, "n_relevant_chunks": 0, "n_total_chunks": 0}
    numbered = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks))
    raw = await call_llm_json(
        client, CHUNK_REL_SYSTEM,
        CHUNK_REL_USER.format(question=question, numbered_chunks=numbered),
    )
    cr = ChunkRelevanceOutput.model_validate(raw)
    valid_indices = [idx for idx in cr.relevant_indices if 0 <= idx < n_total]
    n_relevant = len(valid_indices)
    score = n_relevant / n_total
    return {"context_precision": round(score, 4), "n_relevant_chunks": n_relevant, "n_total_chunks": n_total}


async def compute_context_recall(
    client: AsyncOpenAI, answer: str, chunks: list[str]
) -> dict[str, Any]:
    raw1 = await call_llm_json(
        client, DECOMPOSE_SYSTEM,
        DECOMPOSE_USER.format(answer=answer),
    )
    decomp = DecompOutput.model_validate(raw1)
    statements = decomp.statements

    if not statements:
        return {"context_recall": 0.0, "n_supported": 0, "n_claims": 0}

    context = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks))
    stmt_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    raw2 = await call_llm_json(
        client, RECALL_SYSTEM,
        RECALL_USER.format(context=context, statements=stmt_text),
    )
    rv = RecallVerdicts.model_validate(raw2)

    n_supported = sum(1 for v in rv.verdicts.values() if v == "SUPPORTED")
    n_claims = len(statements)
    score = n_supported / n_claims
    return {"context_recall": round(score, 4), "n_supported": n_supported, "n_claims": n_claims}


# --- Process one QA pair ---


async def process_pair(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pair: dict,
    chunks: list[str],
    article_title: str,
) -> dict:
    async with semaphore:
        q = pair["question"]
        retrieved = retrieve_bm25(pair["answer"], chunks, TOP_K)
        retrieved_chunks = [chunk for _, chunk, _ in retrieved]

        regen_answer, reasoning = await regenerate_answer(client, q, retrieved_chunks)

        cp, cr = await asyncio.gather(
            compute_context_precision(client, q, retrieved_chunks),
            compute_context_recall(client, regen_answer, retrieved_chunks),
        )

        logger.debug(
            f"[{article_title}] {pair['difficulty']} prec={cp['context_precision']:.2f} "
            f"rec={cr['context_recall']:.2f} q={q[:50]}..."
        )

        return {
            "article_id": pair["article_id"],
            "title": article_title,
            "difficulty": pair["difficulty"],
            "question": q,
            "golden_answer": pair["answer"],
            "regen_answer": regen_answer,
            "reasoning_content": reasoning,
            "context_chunks": retrieved_chunks,
            "bm25_scores": [round(s, 3) for _, _, s in retrieved],
            "chunk_indices": [idx for idx, _, _ in retrieved],
            **cp,
            **cr,
        }


# --- Checkpointing ---


def load_processed_article_ids(path: Path) -> set[int]:
    """Load article_ids already in output file for resume."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    ids: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "article_id" in record:
                    ids.add(record["article_id"])
            except json.JSONDecodeError:
                continue
    return ids


def write_jsonl(records: list[dict], path: Path) -> None:
    """Append records to JSONL file."""
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- Main ---


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chunked QA pairs")
    parser.add_argument("n", nargs="?", type=int, default=None, help="Number of articles (default: all)")
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_FILE, help="Output JSONL path")
    args = parser.parse_args()

    output_path = args.output

    # Load tokenizer
    logger.info(f"Loading tokenizer {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # Load all QA pairs, group by article_id
    logger.info(f"Loading QA pairs from {INPUT_FILE}...")
    pairs_by_article: dict[int, list[dict]] = {}
    titles_by_article: dict[int, str] = {}
    with open(INPUT_FILE) as f:
        for line in f:
            p = json.loads(line)
            aid = p["article_id"]
            pairs_by_article.setdefault(aid, []).append(p)
            titles_by_article[aid] = p.get("title", "")

    all_article_ids = sorted(pairs_by_article.keys())
    if args.n is not None:
        all_article_ids = all_article_ids[:args.n]
    logger.info(f"{len(all_article_ids)} articles, {sum(len(pairs_by_article[a]) for a in all_article_ids)} QA pairs")

    # Resume: skip already-processed articles
    processed_ids = load_processed_article_ids(output_path)
    if processed_ids:
        before = len(all_article_ids)
        all_article_ids = [a for a in all_article_ids if a not in processed_ids]
        logger.info(f"Resume: skipped {before - len(all_article_ids)} already-processed, {len(all_article_ids)} remaining")

    if not all_article_ids:
        logger.info("All articles already processed. Nothing to do.")
        return

    # Load articles we need
    logger.info(f"Loading {len(all_article_ids)} articles from wikipedia_en.jsonl...")
    article_texts = load_articles_by_id(set(all_article_ids))
    logger.info(f"Loaded {len(article_texts)} articles")

    client = make_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    total_pairs = 0
    t0 = time.monotonic()

    for batch_start in range(0, len(all_article_ids), BATCH_SIZE):
        batch_ids = all_article_ids[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(all_article_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Batch {batch_num}/{total_batches} ({len(batch_ids)} articles)...")

        # Pre-chunk all articles in batch, then launch ALL pairs concurrently
        all_tasks = []
        for aid in batch_ids:
            text = article_texts.get(aid)
            if not text:
                logger.warning(f"Article {aid} not found in wikipedia_en.jsonl, skipping")
                continue

            chunks = recursive_chunk(text, tokenizer, MAX_CHUNK_TOKENS, SEPARATORS)
            if not chunks:
                logger.warning(f"Article {aid} produced 0 chunks, skipping")
                continue

            title = titles_by_article.get(aid, "")
            pairs = pairs_by_article[aid]
            logger.info(f"  Article {aid} ({title}): {len(chunks)} chunks, {len(pairs)} pairs")

            for p in pairs:
                all_tasks.append(process_pair(client, semaphore, p, chunks, title))

        logger.info(f"  Launching {len(all_tasks)} pairs concurrently (semaphore={MAX_CONCURRENT})...")
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        batch_results = []
        n_failed = 0
        for r in all_results:
            if isinstance(r, Exception):
                n_failed += 1
                logger.error(f"  Failed pair: {r!r}")
            else:
                batch_results.append(r)
        if n_failed:
            logger.warning(f"  {n_failed} pairs failed in batch {batch_num}")

        # Write batch checkpoint
        if batch_results:
            write_jsonl(batch_results, output_path)
            total_pairs += len(batch_results)
            elapsed = time.monotonic() - t0
            rate = total_pairs / elapsed * 60
            logger.info(
                f"  Checkpoint: {total_pairs} pairs written, "
                f"{elapsed:.0f}s elapsed, {rate:.0f} pairs/min"
            )

    elapsed = time.monotonic() - t0
    logger.info(f"Done: {total_pairs} pairs in {elapsed:.0f}s ({total_pairs/elapsed*60:.0f} pairs/min)")


if __name__ == "__main__":
    asyncio.run(main())

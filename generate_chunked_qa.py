"""POC: recursive chunking + BM25 retrieval + context relevance for 1 article."""

import asyncio
import json
import os
import re
import time
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from transformers import AutoTokenizer

# --- Config ---

TOKENIZER_ID = "unsloth/gemma-3-270m-it"
MAX_CHUNK_TOKENS = 512
TOP_K = 3
ARTICLE_ID = 0
DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

SEPARATORS = ["\n\n", "\n", ". ", ", ", " "]


# --- Recursive chunking ---


def recursive_chunk(text: str, tokenizer, max_tokens: int, separators: list[str]) -> list[str]:
    """Recursively split text using separator hierarchy until all chunks <= max_tokens."""
    tok_len = len(tokenizer.encode(text, add_special_tokens=False))
    if tok_len <= max_tokens:
        return [text.strip()] if text.strip() else []

    # Find first separator that actually splits the text
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            # Rejoin with separator (keep it at end of each part except last)
            chunks = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    chunks.append(part + sep)
                else:
                    chunks.append(part)

            # Recursively split any oversized chunks with remaining separators
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
                    # Last resort: hard truncate
                    ids = tokenizer.encode(chunk, add_special_tokens=False)[:max_tokens]
                    result.append(tokenizer.decode(ids).strip())
            return result

    # No separator works — hard truncate
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


# --- Context relevance (chunk-level, RAGAS-style) ---


class ChunkRelevanceOutput(BaseModel):
    reasoning: str = ""
    relevant_indices: list[int]


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


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=3, jitter=1),
    retry=retry_if_exception_type((json.JSONDecodeError, AssertionError)),
    reraise=True,
)
async def call_llm_json(client: AsyncOpenAI, system: str, user: str) -> dict[str, Any]:
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


async def compute_context_relevance(
    client: AsyncOpenAI, question: str, chunks: list[str]
) -> dict[str, Any]:
    n_total = len(chunks)
    if n_total == 0:
        return {"context_relevance": 0.0, "n_relevant": 0, "n_total": 0}

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


# --- Context recall (RAGAS-style: answer claims vs retrieved chunks) ---


class DecompOutput(BaseModel):
    reasoning: str = ""
    statements: list[str]


class RecallVerdicts(BaseModel):
    reasoning: str = ""
    verdicts: dict[str, str]


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


async def compute_context_recall(
    client: AsyncOpenAI, answer: str, chunks: list[str]
) -> dict[str, Any]:
    # Step 1: decompose answer into claims
    raw1 = await call_llm_json(
        client, DECOMPOSE_SYSTEM,
        DECOMPOSE_USER.format(answer=answer),
    )
    decomp = DecompOutput.model_validate(raw1)
    statements = decomp.statements

    if not statements:
        return {"context_recall": 0.0, "n_supported": 0, "n_claims": 0, "statements": [], "verdicts": {}}

    # Step 2: verify claims against retrieved chunks
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
    return {
        "context_recall": round(score, 4),
        "n_supported": n_supported,
        "n_claims": n_claims,
        "statements": statements,
        "verdicts": rv.verdicts,
    }


# --- Answer regeneration from chunks ---

REGEN_SYSTEM = """You answer questions using ONLY the provided context chunks.
Be thorough and detailed, but do NOT include any information not found in the context.
If the context does not contain enough information to fully answer the question, answer
with what is available and note what is missing."""

REGEN_USER = """Context:
{context}

Question:
{question}

Answer the question using ONLY the information in the context above."""


async def regenerate_answer(client: AsyncOpenAI, question: str, chunks: list[str]) -> tuple[str, str]:
    """Returns (answer, reasoning_content)."""
    context = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks))
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


# --- Main ---


async def main():
    t0 = time.monotonic()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # Load article
    print(f"Loading article {ARTICLE_ID}...")
    with open("wikipedia_en.jsonl") as f:
        for line in f:
            art = json.loads(line)
            if art["article_id"] == ARTICLE_ID:
                break
    print(f"  Title: {art['title']}, text length: {len(art['text'])} chars")

    # Chunk
    print(f"\nChunking with max {MAX_CHUNK_TOKENS} tokens...")
    chunks = recursive_chunk(art["text"], tokenizer, MAX_CHUNK_TOKENS, SEPARATORS)
    chunk_tok_lens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in chunks]
    print(f"  {len(chunks)} chunks, token lengths: min={min(chunk_tok_lens)}, max={max(chunk_tok_lens)}, avg={sum(chunk_tok_lens)/len(chunk_tok_lens):.0f}")

    # Load QA pairs for this article
    qa_pairs = []
    with open("qa_pairs.jsonl") as f:
        for line in f:
            p = json.loads(line)
            if p["article_id"] == ARTICLE_ID:
                qa_pairs.append(p)
    print(f"\n{len(qa_pairs)} QA pairs for article {ARTICLE_ID}")

    # Retrieve + evaluate context precision & recall
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)

    async def evaluate_pair(pair: dict) -> dict:
        q = pair["question"]
        # Retrieve by golden answer
        retrieved = retrieve_bm25(pair["answer"], chunks, TOP_K)
        retrieved_chunks = [chunk for _, chunk, _ in retrieved]

        # Regenerate answer from retrieved chunks
        regen_answer, reasoning = await regenerate_answer(client, q, retrieved_chunks)

        # Eval precision (chunks vs question) and recall (regen answer claims vs chunks)
        cp, cr = await asyncio.gather(
            compute_context_relevance(client, q, retrieved_chunks),
            compute_context_recall(client, regen_answer, retrieved_chunks),
        )

        return {
            "question": q,
            "golden_answer": pair["answer"],
            "regen_answer": regen_answer,
            "reasoning_content": reasoning,
            "difficulty": pair["difficulty"],
            "context_chunks": retrieved_chunks,
            "bm25_scores": [round(s, 3) for _, _, s in retrieved],
            "chunk_indices": [idx for idx, _, _ in retrieved],
            **cp,
            **cr,
        }

    print("\nEvaluating all pairs concurrently...")
    results = await asyncio.gather(*[evaluate_pair(p) for p in qa_pairs])
    results = list(results)

    print(f"\n{'='*90}")
    print(f"{'Difficulty':<12} {'Precision':>10} {'Recall':>10} {'Claims':>8}  Question (truncated)")
    print(f"{'='*90}")
    for r in results:
        sup = f"{r['n_supported']}/{r['n_claims']}"
        print(f"{r['difficulty']:<12} {r['context_precision']:>10.3f} {r['context_recall']:>10.3f} {sup:>8}  {r['question'][:55]}...")

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Difficulty':<12} {'Avg Prec':>10} {'Avg Recall':>12}  (n)")
    print(f"{'-'*50}")
    for diff in ["easy", "medium", "hard"]:
        precs = [r["context_precision"] for r in results if r["difficulty"] == diff]
        recs = [r["context_recall"] for r in results if r["difficulty"] == diff]
        if precs:
            print(f"{diff:<12} {sum(precs)/len(precs):>10.3f} {sum(recs)/len(recs):>12.3f}  ({len(precs)})")

    all_prec = [r["context_precision"] for r in results]
    all_rec = [r["context_recall"] for r in results]
    print(f"{'OVERALL':<12} {sum(all_prec)/len(all_prec):>10.3f} {sum(all_rec)/len(all_rec):>12.3f}  ({len(results)})")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Dump full results
    with open("play.py.output.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results written to play.py.output.json")


if __name__ == "__main__":
    asyncio.run(main())

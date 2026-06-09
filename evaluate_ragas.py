"""Evaluate QA pairs using RAGAS-style claim decomposition.

Decomposes both reference (golden) and agent answers into atomic claims,
classifies agent claims as SUPPORTED/CONTRADICTED/UNSUPPORTED against
reference claims, then computes Precision/Recall/F1 per pair.

Reference claims are cached to qa_reference_claims.jsonl for reuse.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

from utils.wikipedia_loader import load_articles_by_id

DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_CONCURRENT = 50
REF_CLAIMS_CACHE_FILE = Path("qa_reference_claims.jsonl")
ARTICLE_CLAIMS_CACHE_FILE = Path("qa_article_claims.jsonl")
LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "evaluate_ragas.log", level=log_level, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}")


class RefClaims(BaseModel):
    article_id: int
    question: str
    claims: list[str]


class ArticleClaims(BaseModel):
    article_id: int
    claims: list[str]


class DecompOutput(BaseModel):
    reasoning: str = ""
    claims: list[str]


class ClassifyOutput(BaseModel):
    reasoning: str
    verdicts: dict[str, str]
    uncovered_ref_indices: list[int]


REF_DECOMPOSE_SYSTEM = "You decompose answers into atomic factual claims. Respond only in json format."

REF_DECOMPOSE_USER = """Break this answer into atomic factual claims. Each claim must be:
1. A single, independently verifiable statement of fact
2. Self-contained (understandable without additional context)
3. Concise (one sentence per claim)

Answer: {answer}

Respond in JSON format: {{"reasoning": "brief note", "claims": ["claim1", "claim2", ...]}}"""

CLASSIFY_SYSTEM = """You classify a list of claims against reference claim ground truth.
SUPPORTED means factual content matches (exact wording not required).
CONTRADICTED means factual content disagrees on a specific point.
UNSUPPORTED means the claim goes beyond what reference claims state (hallucination).
Be strict but fair. Respond only in json format."""

CLASSIFY_USER = """Reference claims:
{ref_claims}

Agent claims to classify:
{agent_claims}

For each agent claim, classify as SUPPORTED, CONTRADICTED, or UNSUPPORTED.
Also list indices (0-based) of reference claims NOT covered by any agent claim.

Respond in JSON format: {{"reasoning": "...", "verdicts": {{"claim text": "SUPPORTED"|"CONTRADICTED"|"UNSUPPORTED"}}, "uncovered_ref_indices": [0, 2]}}"""


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)


def load_json_or_jsonl(path: Path) -> list[dict]:
    raw = path.read_text().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def ref_key(article_id: int, question: str) -> str:
    return f"{article_id}::{question}"


def load_ref_claims_cache() -> dict[str, RefClaims]:
    if not REF_CLAIMS_CACHE_FILE.exists():
        return {}
    cache: dict[str, RefClaims] = {}
    for line in REF_CLAIMS_CACHE_FILE.read_text().strip().splitlines():
        if not line.strip():
            continue
        rc = RefClaims.model_validate(json.loads(line))
        cache[ref_key(rc.article_id, rc.question)] = rc
    return cache


def save_ref_claims_cache(cache: dict[str, RefClaims]) -> None:
    lines = [
        json.dumps({"article_id": rc.article_id, "question": rc.question, "claims": rc.claims}, ensure_ascii=False)
        for rc in cache.values()
    ]
    REF_CLAIMS_CACHE_FILE.write_text("\n".join(lines))


def load_article_claims_cache() -> dict[int, ArticleClaims]:
    if not ARTICLE_CLAIMS_CACHE_FILE.exists():
        return {}
    cache: dict[int, ArticleClaims] = {}
    for line in ARTICLE_CLAIMS_CACHE_FILE.read_text().strip().splitlines():
        if not line.strip():
            continue
        ac = ArticleClaims.model_validate(json.loads(line))
        cache[ac.article_id] = ac
    return cache


def save_article_claims_cache(cache: dict[int, ArticleClaims]) -> None:
    lines = [
        json.dumps({"article_id": aid, "claims": ac.claims}, ensure_ascii=False)
        for aid, ac in cache.items()
    ]
    ARTICLE_CLAIMS_CACHE_FILE.write_text("\n".join(lines))


async def decompose_reference(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pair: dict,
) -> RefClaims | None:
    async with semaphore:
        answer = pair["answer"]
        t0 = time.monotonic()

        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": REF_DECOMPOSE_SYSTEM},
                {"role": "user", "content": REF_DECOMPOSE_USER.format(answer=answer)},
            ],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        if not raw:
            logger.warning(f"[{pair['title']}] empty ref decomposition response")
            return None

        data = DecompOutput.model_validate(json.loads(raw))
        elapsed = time.monotonic() - t0
        logger.info(f"[{pair['title']}] ref decomposed: {len(data.claims)} claims in {elapsed:.1f}s")
        return RefClaims(article_id=pair["article_id"], question=pair["question"], claims=data.claims)


async def decompose_article(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    article_id: int,
    article_text: str,
) -> ArticleClaims | None:
    async with semaphore:
        t0 = time.monotonic()
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You extract all key factual claims from Wikipedia articles. Respond only in json format."},
                {"role": "user", "content": f"Extract all key factual claims from this Wikipedia article. Each claim must be a single, independently verifiable statement of fact. Focus on the most important facts.\n\nArticle:\n{article_text}\n\nRespond in JSON format: {{\"reasoning\": \"brief note\", \"claims\": [\"claim1\", \"claim2\", ...]}}"},
            ],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        if not raw:
            logger.warning(f"[article_id={article_id}] empty article decomposition response")
            return None

        data = DecompOutput.model_validate(json.loads(raw))
        elapsed = time.monotonic() - t0
        logger.info(f"[article_id={article_id}] article decomposed: {len(data.claims)} claims in {elapsed:.1f}s")
        return ArticleClaims(article_id=article_id, claims=data.claims)


async def evaluate_pair(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pair: dict,
    ref_claims: list[str] | None = None,
    article_claims: list[str] | None = None,
) -> dict | None:
    async with semaphore:
        answer = pair.get("model_answer", pair["answer"])
        baseline = ref_claims if ref_claims is not None else article_claims
        assert baseline is not None, "must provide ref_claims or article_claims"
        t0 = time.monotonic()

        # Call 1: decompose agent answer into atomic claims
        resp1 = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": REF_DECOMPOSE_SYSTEM},
                {"role": "user", "content": REF_DECOMPOSE_USER.format(answer=answer)},
            ],
            response_format={"type": "json_object"},
        )

        raw1 = resp1.choices[0].message.content
        if not raw1:
            logger.warning(f"[{pair['title']}] empty decomposition response")
            return None

        agent_claims = DecompOutput.model_validate(json.loads(raw1)).claims

        # Call 2: classify agent claims against baseline claims
        eval_mode = "reference" if ref_claims is not None else "article"
        baseline_text = "\n".join(f"{i}. {c}" for i, c in enumerate(baseline))
        agent_claims_text = "\n".join(f"{i}. {c}" for i, c in enumerate(agent_claims))

        resp2 = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": CLASSIFY_USER.format(ref_claims=baseline_text, agent_claims=agent_claims_text)},
            ],
            response_format={"type": "json_object"},
        )

        raw2 = resp2.choices[0].message.content
        if not raw2:
            logger.warning(f"[{pair['title']}] empty classification response")
            return None

        data = ClassifyOutput.model_validate(json.loads(raw2))
        elapsed = time.monotonic() - t0

        supported = sum(1 for v in data.verdicts.values() if v == "SUPPORTED")
        contradicted = sum(1 for v in data.verdicts.values() if v == "CONTRADICTED")
        unsupported = sum(1 for v in data.verdicts.values() if v == "UNSUPPORTED")
        uncovered = len(data.uncovered_ref_indices)

        tp = supported
        fp = contradicted + unsupported
        fn = uncovered

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result = {
            **pair,
            "eval_mode": eval_mode,
            "reference_claims": baseline,
            "agent_claims": agent_claims,
            "supported": supported,
            "contradicted": contradicted,
            "unsupported": unsupported,
            "uncovered": uncovered,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "judge_reasoning": data.reasoning,
            "eval_time_s": round(elapsed, 1),
        }

        logger.info(
            f"[{pair['title']}] F1={result['f1']:.2f} P={result['precision']:.2f} R={result['recall']:.2f} "
            f"(S={supported} C={contradicted} U={unsupported} UC={uncovered}) in {elapsed:.1f}s"
        )
        return result


async def main(
    input_path: Path,
    output_path: Path,
    reference_path: Path,
) -> None:
    eval_pairs = load_json_or_jsonl(input_path)
    logger.info(f"Loaded {len(eval_pairs)} pairs to evaluate from {input_path}")

    client = make_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Split pairs by eval mode
    ref_pairs: list[dict] = []
    art_pairs: list[dict] = []
    for p in eval_pairs:
        if "model_answer" in p:
            ref_pairs.append(p)
        else:
            art_pairs.append(p)

    logger.info(f"Reference-based eval: {len(ref_pairs)} pairs (has model_answer)")
    logger.info(f"Article-based eval:   {len(art_pairs)} pairs (evaluating against Wikipedia text)")

    # Phase 1a: ensure reference claims cached (for ref_pairs)
    ref_cache = load_ref_claims_cache()
    if ref_pairs:
        missing: list[dict] = []
        for p in ref_pairs:
            key = ref_key(p["article_id"], p["question"])
            if key not in ref_cache:
                missing.append(p)

        if missing:
            # Load reference pairs from reference_path for decomposition
            ref_source = load_json_or_jsonl(reference_path)
            ref_source_map: dict[str, dict] = {}
            for rp in ref_source:
                ref_source_map[ref_key(rp["article_id"], rp["question"])] = rp

            to_decompose: list[dict] = []
            for p in missing:
                src = ref_source_map.get(ref_key(p["article_id"], p["question"]))
                if src:
                    to_decompose.append(src)
                else:
                    logger.warning(f"[{p['title']}] no reference source found for article_id={p['article_id']}")

            if to_decompose:
                logger.info(f"Decomposing {len(to_decompose)} uncached reference answers...")
                t0 = time.monotonic()
                coros = [decompose_reference(client, semaphore, p) for p in to_decompose]
                ref_results = await asyncio.gather(*coros, return_exceptions=True)
                for r in ref_results:
                    if isinstance(r, Exception):
                        logger.error(f"Ref decomposition error: {r}")
                        continue
                    if r:
                        ref_cache[ref_key(r.article_id, r.question)] = r
                save_ref_claims_cache(ref_cache)
                logger.info(f"Cached reference decompositions in {time.monotonic()-t0:.1f}s")
        else:
            logger.info(f"All {len(ref_pairs)} reference claims already cached")

    # Phase 1b: ensure article claims cached (for art_pairs)
    art_cache = load_article_claims_cache()
    if art_pairs:
        article_ids_needed = {p["article_id"] for p in art_pairs if p["article_id"] not in art_cache}
        if article_ids_needed:
            logger.info(f"Fetching {len(article_ids_needed)} articles for claim decomposition...")
            article_texts = load_articles_by_id(article_ids_needed)

            missing_articles = [
                (aid, text) for aid, text in article_texts.items()
                if aid not in art_cache and text
            ]
            if missing_articles:
                logger.info(f"Decomposing {len(missing_articles)} uncached articles...")
                t0 = time.monotonic()
                coros = [decompose_article(client, semaphore, aid, text) for aid, text in missing_articles]
                art_results = await asyncio.gather(*coros, return_exceptions=True)
                for r in art_results:
                    if isinstance(r, Exception):
                        logger.error(f"Article decomposition error: {r}")
                        continue
                    if r:
                        art_cache[r.article_id] = r
                save_article_claims_cache(art_cache)
                logger.info(f"Cached article decompositions in {time.monotonic()-t0:.1f}s")
        else:
            logger.info(f"All article claims already cached")

    # Phase 2: evaluate all pairs (with resume support)
    completed_keys: set[str] = set()
    results: list[dict] = []
    if output_path.exists():
        try:
            for line in output_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                completed_keys.add(ref_key(r["article_id"], r["question"]))
                results.append(r)
            logger.info(f"Resuming: {len(results)} already evaluated, reusing")
        except Exception:
            logger.warning("Could not load existing output, starting fresh")
            output_path.write_text("")
            results = []

    pending = [p for p in eval_pairs if ref_key(p["article_id"], p["question"]) not in completed_keys]
    if len(pending) < len(eval_pairs):
        logger.info(f"Skipping {len(eval_pairs) - len(pending)} already-evaluated pairs, {len(pending)} remaining")
    if not pending:
        logger.info("All pairs already evaluated")
    t0 = time.monotonic()

    coros = []
    for p in pending:
        if "model_answer" in p:
            rc = ref_cache.get(ref_key(p["article_id"], p["question"]))
            if not rc:
                logger.warning(f"[{p['title']}] no reference claims found, skipping")
                continue
            coros.append(evaluate_pair(client, semaphore, p, ref_claims=rc.claims))
        else:
            ac = art_cache.get(p["article_id"])
            if not ac:
                logger.warning(f"[{p['title']}] no article claims for article_id={p['article_id']}, skipping")
                continue
            coros.append(evaluate_pair(client, semaphore, p, article_claims=ac.claims))

    done_count = 0
    for fut in asyncio.as_completed(coros):
        try:
            r = await fut
        except Exception as e:
            logger.error(f"Eval error: {e}")
            done_count += 1
            continue
        done_count += 1
        if r:
            results.append(r)
        if done_count % MAX_CONCURRENT == 0 or done_count == len(coros):
            output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
            logger.info(f"Progress: {len(results)}/{len(eval_pairs)} evaluated, {len(coros)-done_count} remaining ({time.monotonic()-t0:.1f}s)")

    # Summary
    elapsed = time.monotonic() - t0
    if results:
        ref_results = [r for r in results if r["eval_mode"] == "reference"]
        art_results = [r for r in results if r["eval_mode"] == "article"]

        def avg(lst, key):
            return sum(r[key] for r in lst) / len(lst) if lst else 0.0

        logger.info(f"=== Summary ({len(results)} pairs, {elapsed:.1f}s) ===")
        if ref_results:
            logger.info(f"Reference-based ({len(ref_results)}): P={avg(ref_results, 'precision'):.3f} R={avg(ref_results, 'recall'):.3f} F1={avg(ref_results, 'f1'):.3f}")
        if art_results:
            logger.info(f"Article-based ({len(art_results)}):   P={avg(art_results, 'precision'):.3f} R={avg(art_results, 'recall'):.3f} F1={avg(art_results, 'f1'):.3f}")
        logger.info(f"Overall: P={avg(results, 'precision'):.3f} R={avg(results, 'recall'):.3f} F1={avg(results, 'f1'):.3f}")
        logger.info(f"Output:    {output_path}")
    else:
        logger.warning("No results to summarize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA pairs with RAGAS-style claim decomposition")
    parser.add_argument("input", nargs="?", default="qa_pairs.jsonl", help="Input JSON/JSONL file to evaluate")
    parser.add_argument(
        "--reference", default="qa_pairs.jsonl", help="Reference (golden) QA pairs file (default: qa_pairs.jsonl)"
    )
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file (default: <input_stem>_ragas_eval.jsonl)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_ragas_eval").with_suffix(".jsonl")
    ref_path = Path(args.reference)
    asyncio.run(main(in_path, out_path, ref_path))

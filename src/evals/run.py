"""Run the full RAGAS eval pipeline: single question or batch."""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from src.evals.classify import ClassifyOutput, classify_claims
from src.evals.decompose import decompose_answer
from src.evals.scoring import Score, compute_score


@dataclass
class EvalResult:
    ref_claims: list[str]
    agent_claims: list[str]
    classification: ClassifyOutput
    score: Score


async def evaluate_single(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    ref_answer: str,
    agent_answer: str,
    model: str,
    question: str = "",
) -> EvalResult:
    """Decompose both answers, classify, and score."""
    ref_claims, agent_claims = await asyncio.gather(
        decompose_answer(client, semaphore, ref_answer, model, question=question),
        decompose_answer(client, semaphore, agent_answer, model, question=question),
    )

    if ref_claims is None:
        raise ValueError("Failed to decompose reference answer")
    if agent_claims is None:
        raise ValueError("Failed to decompose agent answer")

    classification = await classify_claims(client, semaphore, agent_claims, ref_claims, model)
    if classification is None:
        raise ValueError("Failed to classify claims")

    return EvalResult(
        ref_claims=ref_claims,
        agent_claims=agent_claims,
        classification=classification,
        score=compute_score(classification),
    )


def _load_json_or_jsonl(path: Path) -> list[dict]:
    raw = path.read_text().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.split("\n") if line.strip()]


def _pair_key(article_id: int, question: str) -> str:
    return f"{article_id}::{question}"


def _strip_reasoning(text: str) -> str:
    return re.sub(r"<reasoning>.*?</reasoning>", "", text, count=1, flags=re.DOTALL).strip()


async def evaluate_all(
    input_path: Path,
    output_path: Path,
    reference_path: Path = Path("qa_pairs_chunked_test.jsonl"),
    model: str = "deepseek-v4-flash",
    base_url: str = "https://api.deepseek.com",
    max_concurrent: int = 50,
    overwrite: bool = False,
) -> list[dict]:
    """Evaluate all pairs from input against reference, write results to output."""
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}. Pass overwrite=True to replace.")
        output_path.unlink()

    input_pairs = _load_json_or_jsonl(input_path)
    ref_pairs = _load_json_or_jsonl(reference_path)
    logger.info(f"Loaded {len(input_pairs)} input pairs, {len(ref_pairs)} reference pairs")

    # Build ref lookup by article_id::question
    ref_map: dict[str, dict] = {}
    for rp in ref_pairs:
        ref_map[_pair_key(rp["article_id"], rp["question"])] = rp

    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=base_url)
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[dict] = []

    pairs_to_eval: list[dict] = []
    coros = []
    for pair in input_pairs:
        key = _pair_key(pair["article_id"], pair["question"])
        ref = ref_map.get(key)
        if not ref:
            logger.warning(f"[{pair.get('title', key)}] no reference found, skipping")
            continue
        agent_answer = _strip_reasoning(pair["model_answer"])
        pairs_to_eval.append(pair)
        coros.append(evaluate_single(client, semaphore, ref["regen_answer"], agent_answer, model, question=pair["question"]))

    eval_results = await asyncio.gather(*coros, return_exceptions=True)

    for pair, result in zip(pairs_to_eval, eval_results):
        if isinstance(result, Exception):
            logger.error(f"[{pair.get('title', '')}] eval failed: {result}")
            continue
        results.append({
            **pair,
            "ref_claims": result.ref_claims,
            "agent_claims": result.agent_claims,
            "verdicts": result.classification.verdicts,
            "uncovered_ref_indices": result.classification.uncovered_ref_indices,
            "judge_reasoning": result.classification.reasoning,
            "supported": result.score.supported,
            "contradicted": result.score.contradicted,
            "unsupported": result.score.unsupported,
            "uncovered": result.score.uncovered,
            "precision": result.score.precision,
            "recall": result.score.recall,
            "f1": result.score.f1,
        })

    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))

    # Summary
    if results:
        avg_p = sum(r["precision"] for r in results) / len(results)
        avg_r = sum(r["recall"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        logger.info(f"=== Summary ({len(results)} pairs) === P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")
        logger.info(f"Output: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS-style claim-based evaluation")
    parser.add_argument("input", help="Input JSON/JSONL file with model_answer field")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file (default: <input_stem>_ragas_eval.jsonl)")
    parser.add_argument("--reference", default="qa_pairs_chunked_test.jsonl", help="Reference QA pairs file")
    parser.add_argument("--model", default="deepseek-v4-flash", help="LLM judge model")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="LLM API base URL")
    parser.add_argument("--max-concurrent", type=int, default=50, help="Max concurrent LLM calls")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_ragas_eval").with_suffix(".jsonl")

    asyncio.run(evaluate_all(
        input_path=in_path,
        output_path=out_path,
        reference_path=Path(args.reference),
        model=args.model,
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        overwrite=args.overwrite,
    ))

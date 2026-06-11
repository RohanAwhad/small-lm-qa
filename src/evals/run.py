"""Run the full RAGAS eval pipeline for a single question."""

import asyncio
from dataclasses import dataclass

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
) -> EvalResult:
    """Decompose both answers, classify, and score."""
    ref_claims, agent_claims = await asyncio.gather(
        decompose_answer(client, semaphore, ref_answer, model),
        decompose_answer(client, semaphore, agent_answer, model),
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

"""Classify agent claims against reference claims via LLM."""

import asyncio
import json

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


class ClassifyOutput(BaseModel):
    reasoning: str
    verdicts: dict[str, str]
    uncovered_ref_indices: list[int]


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


@retry(stop=stop_after_attempt(20), wait=wait_exponential_jitter(max=30))
async def classify_claims(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    agent_claims: list[str],
    ref_claims: list[str],
    model: str,
) -> ClassifyOutput | None:
    """Classify each agent claim against reference claims."""
    async with semaphore:
        ref_text = "\n".join(f"{i}. {c}" for i, c in enumerate(ref_claims))
        agent_text = "\n".join(f"{i}. {c}" for i, c in enumerate(agent_claims))

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": CLASSIFY_USER.format(ref_claims=ref_text, agent_claims=agent_text)},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        if not raw:
            return None
        return ClassifyOutput.model_validate(json.loads(raw))

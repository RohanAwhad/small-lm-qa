"""Decompose answers into atomic factual claims via LLM."""

import asyncio
import json

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


class DecompOutput(BaseModel):
    reasoning: str = ""
    claims: list[str]



REF_DECOMPOSE_SYSTEM = "You decompose answers into atomic factual claims. Respond only in json format."

REF_DECOMPOSE_USER = """Break this answer into atomic factual claims. Each claim must be:
1. A single, independently verifiable statement of fact
2. Self-contained (understandable without additional context)
3. Concise (one sentence per claim)

Question: {question}
Answer: {answer}

Respond in JSON format: {{"reasoning": "brief note", "claims": ["claim1", "claim2", ...]}}"""



@retry(stop=stop_after_attempt(20), wait=wait_exponential_jitter(max=30))
async def decompose_answer(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    answer: str,
    model: str,
    question: str = "",
) -> list[str] | None:
    """Decompose any answer text into atomic claims."""
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REF_DECOMPOSE_SYSTEM},
                {"role": "user", "content": REF_DECOMPOSE_USER.format(question=question, answer=answer)},
            ],
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        raw = resp.choices[0].message.content
        if not raw:
            return None
        return DecompOutput.model_validate(json.loads(raw)).claims




"""vLLM OpenAI-compatible API inference engine."""

import asyncio
import os
import time

from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from src.generation.constants import DEFAULT_SAMPLING_PARAMS


@retry(stop=stop_after_attempt(20), wait=wait_exponential_jitter(max=30))
async def _generate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    messages: list[dict[str, str]],
    idx: int,
) -> tuple[int, str]:
    params = DEFAULT_SAMPLING_PARAMS
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
        )
        return idx, resp.choices[0].message.content


async def _run_all(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    messages_list: list[list[dict[str, str]]],
    model: str,
) -> list[str]:
    t0 = time.monotonic()
    results: list[str | None] = [None] * len(messages_list)
    done_count = 0
    coros = [_generate_one(client, semaphore, model, messages_list[i], i) for i in range(len(messages_list))]
    for coro in asyncio.as_completed(coros):
        idx, answer = await coro
        results[idx] = answer
        done_count += 1
        if done_count % 50 == 0 or done_count == len(messages_list):
            logger.info(f"Progress: {done_count}/{len(messages_list)} ({time.monotonic() - t0:.1f}s)")
    elapsed = time.monotonic() - t0
    logger.info(f"vLLM inference done in {elapsed:.1f}s")
    return results  # type: ignore[return-value]


def generate(
    messages_list: list[list[dict[str, str]]],
    model: str,
    max_concurrent: int,
    base_url: str,
) -> list[str]:
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    return asyncio.run(_run_all(client, semaphore, messages_list, model))

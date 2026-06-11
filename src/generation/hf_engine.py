"""HF Transformers batch inference engine."""

import time

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.constants import DEFAULT_SAMPLING_PARAMS


def generate(
    messages_list: list[list[dict[str, str]]],
    model_id: str,
    batch_size: int,
) -> list[str]:
    params = DEFAULT_SAMPLING_PARAMS

    # Load model
    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    logger.info(f"Model loaded on {model.device}")

    # Build prompts
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    max_input_len = getattr(model.config, "max_position_embeddings", 8192) - params.max_new_tokens

    # Batched inference
    all_answers: list[str] = []
    t0 = time.monotonic()
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        ).to(model.device)

        bt0 = time.monotonic()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                temperature=params.temperature,
            )
        bt = time.monotonic() - bt0

        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            all_answers.append(text)

        logger.info(
            f"Batch {batch_start // batch_size + 1}: "
            f"{len(all_answers)}/{len(prompts)} done ({bt:.1f}s, "
            f"{len(batch_prompts) / bt:.1f} pairs/s)"
        )

    elapsed = time.monotonic() - t0
    logger.info(f"HF inference done in {elapsed:.1f}s")
    return all_answers

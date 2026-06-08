"""Generate answers for QA pairs using HF Transformers on GPU.

Drop-in replacement for generate_gemma_answers.py but runs locally via HF
instead of Ollama. Designed for fast batch inference on a GPU node.

Output schema matches generate_gemma_answers.py: adds model_answer + model fields.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 1024
BATCH_SIZE = 64
LOG_DIR = Path("logs")

# Matches train_hf_gemma3.py system prompt
SYSTEM_PROMPT = "Answer the question using the provided context."

LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "generate_hf_answers.log", level="DEBUG", rotation="10 MB")


def build_prompt(tokenizer, question: str, context: str) -> str:
    """Build chat prompt matching train_hf_gemma3.py format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
) -> list[str]:
    """Generate answers for a batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings - MAX_NEW_TOKENS,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode only the generated tokens (skip input)
    answers = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        answers.append(text)
    return answers


def main(input_path: Path, output_path: Path, model_id: str, batch_size: int) -> None:
    # Load QA pairs
    raw = input_path.read_text().strip()
    if raw.startswith("["):
        pairs = json.loads(raw)
    else:
        pairs = [json.loads(l) for l in raw.splitlines() if l.strip()]
    logger.info(f"Loaded {len(pairs)} QA pairs from {input_path}")

    # Detect chunked input (has context_chunks field)
    has_chunks = "context_chunks" in pairs[0] if pairs else False
    if has_chunks:
        logger.info("Detected chunked input — using context_chunks (matches training format)")
    else:
        logger.info("No context_chunks found — falling back to full article text")
        from utils.wikipedia_loader import load_articles_by_id
        article_ids = {p["article_id"] for p in pairs}
        logger.info(f"Loading {len(article_ids)} unique articles...")
        article_texts = load_articles_by_id(article_ids)
        logger.info(f"Loaded {len(article_texts)} article texts")

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
    valid_pairs = []
    prompts = []
    for pair in pairs:
        if has_chunks:
            context = "\n\n".join(pair["context_chunks"])
        else:
            context = article_texts.get(pair["article_id"], "")
            if not context:
                logger.warning(f"[{pair['title']}] article_id={pair['article_id']} not found, skipping")
                continue
        valid_pairs.append(pair)
        prompts.append(build_prompt(tokenizer, pair["question"], context))

    logger.info(f"Processing {len(valid_pairs)} pairs in batches of {batch_size}")

    # Model name for output (use dir name for local checkpoints)
    model_label = Path(model_id).name if "/" not in model_id or model_id.startswith(".") else model_id

    t0 = time.monotonic()
    written = 0
    with open(output_path, "w") as f:
        for batch_start in range(0, len(valid_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_pairs))
            batch_prompts = prompts[batch_start:batch_end]
            batch_pairs = valid_pairs[batch_start:batch_end]

            bt0 = time.monotonic()
            answers = generate_batch(model, tokenizer, batch_prompts)
            bt = time.monotonic() - bt0

            for pair, answer in zip(batch_pairs, answers):
                # Normalize golden_answer -> answer for evaluate_ragas.py compatibility
                out = {**pair, "model_answer": answer, "model": model_label}
                if "golden_answer" in out and "answer" not in out:
                    out["answer"] = out["golden_answer"]
                result = out
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

            logger.info(
                f"Batch {batch_start // batch_size + 1}: "
                f"{written}/{len(valid_pairs)} done ({bt:.1f}s, "
                f"{len(batch_prompts) / bt:.1f} pairs/s)"
            )

    elapsed = time.monotonic() - t0
    logger.info(f"Done. {written} answers written to {output_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers via HF Transformers on GPU")
    parser.add_argument("input", nargs="?", default="qa_test.json", help="Input JSON/JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file")
    parser.add_argument("-m", "--model", default="unsloth/gemma-3-270m-it", help="HF model ID or local checkpoint path")
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_hf").with_suffix(".jsonl")
    main(in_path, out_path, args.model, args.batch_size)

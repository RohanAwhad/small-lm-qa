"""Generate model answers for QA pairs via HF Transformers or vLLM API."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from src.generation.utils import generate_messages

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_DIR / "generate_answers.log", level="DEBUG", rotation="10 MB")


class QAPair(BaseModel):
    article_id: int
    title: str
    difficulty: str
    question: str
    golden_answer: str
    regen_answer: str
    reasoning_content: str
    context_chunks: list[str]
    bm25_scores: list[float]
    chunk_indices: list[int]
    context_precision: float
    n_relevant_chunks: int
    n_total_chunks: int
    context_recall: float
    n_supported: int
    n_claims: int


def _load_qa_pairs(path: Path) -> list[QAPair]:
    raw = path.read_text().strip()
    if raw.startswith("["):
        items = json.loads(raw)
    else:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return [QAPair(**item) for item in items]


def run(
    input_path: Path,
    output_path: Path,
    engine: str,
    model: str,
    batch_size: int,
    base_url: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace.")

    # 1. Load and validate input
    pairs = _load_qa_pairs(input_path)
    logger.info(f"Loaded {len(pairs)} QA pairs from {input_path}")

    # 2. Build messages per pair
    messages_list: list[list[dict[str, str]]] = []
    for pair in pairs:
        ctx = "\n\n".join(pair.context_chunks)
        messages_list.append(generate_messages(ctx, pair.question))

    logger.info(f"Processing {len(pairs)} pairs via {engine} (batch_size={batch_size})")

    # 3. Inference (engine-specific)
    if engine == "hf":
        from src.generation.hf_engine import generate
        answers = generate(messages_list, model, batch_size)
    elif engine == "vllm":
        from src.generation.vllm_engine import generate
        answers = generate(messages_list, model, batch_size, base_url)

    # 4. Write output
    model_label = Path(model).name if "/" not in model or model.startswith(".") else model
    with open(output_path, "w") as f:
        for pair, answer in zip(pairs, answers):
            out = {**pair.model_dump(), "model_answer": answer, "model": model_label}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info(f"Done. {len(answers)} answers written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model answers for QA pairs")
    parser.add_argument("input", help="Input JSON/JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL (default: <input>_gen.jsonl)")
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf", help="Inference engine (default: hf)")
    parser.add_argument("-m", "--model", default="unsloth/gemma-3-270m-it", help="Model ID or checkpoint path")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size (hf) / max concurrent (vllm)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM API base URL (vllm engine only)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_gen").with_suffix(".jsonl")
    run(in_path, out_path, args.engine, args.model, args.batch_size, args.base_url, args.overwrite)

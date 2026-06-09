"""Build DPO training pairs from judged rollouts.

Reads judged rollouts JSONL, filters out all_bad/all_good/invalid,
extracts chosen/rejected responses, and outputs in trl DPOTrainer format.

Output format (per line):
{
  "prompt": [{"role": "system", ...}, {"role": "user", ...}],
  "chosen": [{"role": "assistant", "content": "<chosen response>"}],
  "rejected": [{"role": "assistant", "content": "<rejected response>"}]
}

Usage:
    uv run python build_dpo_pairs.py dpo_rollouts_judged.jsonl -o dpo_pairs_train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

SYSTEM_PROMPT = "Answer the question using the provided context."


def main(input_path: Path, output_path: Path) -> None:
    records = [json.loads(l) for l in input_path.read_text().strip().splitlines() if l.strip()]
    logger.info(f"Loaded {len(records)} judged records from {input_path}")

    n_written = 0
    n_all_bad = 0
    n_all_good = 0
    n_invalid = 0

    with open(output_path, "w") as f:
        for r in records:
            j = r["judgment"]

            if j.get("all_bad"):
                n_all_bad += 1
                continue
            if j.get("all_good"):
                n_all_good += 1
                continue

            best_idx = j.get("best_idx")
            worst_idx = j.get("worst_idx")

            if best_idx is None or worst_idx is None or best_idx == worst_idx:
                n_invalid += 1
                continue

            rollouts = r["rollouts"]
            if best_idx >= len(rollouts) or worst_idx >= len(rollouts):
                n_invalid += 1
                continue

            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{r['context']}\n\nQuestion: {r['question']}"},
            ]
            chosen = [{"role": "assistant", "content": rollouts[best_idx]}]
            rejected = [{"role": "assistant", "content": rollouts[worst_idx]}]

            pair = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        f"Done. {n_written} DPO pairs written to {output_path} "
        f"(all_bad={n_all_bad}, all_good={n_all_good}, invalid={n_invalid})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DPO pairs from judged rollouts")
    parser.add_argument("input", help="Input JSONL with judged rollouts")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL with DPO pairs")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else Path("dpo_pairs_train.jsonl")
    main(in_path, out_path)

"""Split JSONL dataset into train/test. Test capped at N per difficulty.

Usage:
  uv run python split_train_test.py                          # split qa_pairs.jsonl
  uv run python split_train_test.py qa_pairs_chunked.jsonl   # split chunked QA
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("qa_pairs_chunked.jsonl")
TEST_PER_DIFFICULTY = 200
SEED = 42


def main() -> None:
    random.seed(SEED)

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    stem = input_path.stem  # e.g. "qa_pairs_chunked"
    suffix = input_path.suffix  # e.g. ".jsonl"
    train_out = Path(f"{stem}_train{suffix}")
    test_out = Path(f"{stem}_test{suffix}")

    # Load all pairs
    pairs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs from {input_path}")

    # Group by difficulty, filter out unknown
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        d = p.get("difficulty", "unknown")
        if d == "unknown":
            continue
        by_diff[d].append(p)

    test, train = [], []
    for diff in ("easy", "medium", "hard"):
        pool = by_diff.get(diff, [])
        random.shuffle(pool)
        n_test = min(TEST_PER_DIFFICULTY, len(pool))
        test.extend(pool[:n_test])
        train.extend(pool[n_test:])
        print(f"{diff}: {len(pool)} total -> {n_test} test, {len(pool)-n_test} train")

    random.shuffle(test)
    random.shuffle(train)

    with open(train_out, "w") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(test_out, "w") as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nTest:  {len(test)} -> {test_out}")
    print(f"Train: {len(train)} -> {train_out}")


if __name__ == "__main__":
    main()

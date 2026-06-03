"""Split qa_pairs.jsonl into train/test JSONs. Test capped at 200 per difficulty."""

import json
import random
from collections import defaultdict
from pathlib import Path

INPUT = Path("qa_pairs.jsonl")
TEST_PER_DIFFICULTY = 200
TEST_OUT = Path("qa_test.json")
TRAIN_OUT = Path("qa_train.json")
SEED = 42


def main() -> None:
    random.seed(SEED)

    # Load all pairs
    lines = Path(INPUT).read_text().strip().splitlines()
    pairs = [json.loads(l) for l in lines]

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

    TEST_OUT.write_text(json.dumps(test, indent=2, ensure_ascii=False))
    TRAIN_OUT.write_text(json.dumps(train, indent=2, ensure_ascii=False))

    print(f"\nTest:  {len(test)} questions -> {TEST_OUT}")
    print(f"Train: {len(train)} questions -> {TRAIN_OUT}")


if __name__ == "__main__":
    main()

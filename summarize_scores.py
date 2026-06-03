"""Pretty-print RAGAS evaluation scores grouped by difficulty level."""

import json
import sys
from collections import defaultdict
from pathlib import Path

SCORE_FIELDS = ["f1", "precision", "recall"]
COUNT_FIELDS = ["supported", "contradicted", "unsupported", "uncovered"]

METRIC_DEFS = {
    "f1": "harmonic mean of P and R",
    "precision": "S / (S+C+U) — accuracy of claims",
    "recall": "S / (S+UC) — completeness of answer",
    "supported": "agent claims matching ref (TP)",
    "contradicted": "agent claims contradicting ref",
    "unsupported": "agent claims not in ref (hallucination)",
    "uncovered": "ref claims not addressed (missed)",
}


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().strip().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def avg(items: list[dict], field: str) -> float:
    vals = [r[field] for r in items if field in r]
    return sum(vals) / len(vals) if vals else 0.0


def print_table(
    groups: dict[str, list[dict]],
    fields: list[str],
    all_records: list[dict],
    col_w: int = 12,
    decimals: int = 3,
) -> None:
    label_w = max(len(d) for d in groups if d != "unknown") + 2
    nf = len(fields)
    header = f"{'':{label_w}}" + "".join(f"{f:>{col_w}}" for f in fields)
    divider = f"{'':{label_w}}" + "".join("─" * col_w for _ in range(nf))
    print(header)
    print(divider)

    for difficulty in ("easy", "medium", "hard"):
        if difficulty not in groups:
            continue
        items = groups[difficulty]
        vals = "".join(f"{avg(items, f):{col_w}.{decimals}f}" for f in fields)
        print(f"{difficulty:<{label_w}}{vals}  (n={len(items):>5})")

    print(divider)
    overall = "".join(f"{avg(all_records, f):{col_w}.{decimals}f}" for f in fields)
    print(f"{'overall':<{label_w}}{overall}  (n={len(all_records):>5})")
    print()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("qa_pairs_evaluated.jsonl")
    records = load_records(path)

    if not records:
        print("No records found.")
        return

    groups: dict[str, list[dict]] = defaultdict(list)
    eval_modes: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get("difficulty", "unknown")].append(r)
        eval_modes[r.get("eval_mode", "unknown")].append(r)

    print(f"File: {path}  ({len(records)} pairs)")
    for mode, items in sorted(eval_modes.items()):
        print(f"  {mode}: {len(items)} pairs")
    print()

    print("=== Scores ===")
    print_table(groups, SCORE_FIELDS, records, 12, 3)
    for f in SCORE_FIELDS:
        print(f"  {f:<16} {METRIC_DEFS.get(f, '')}")
    print()

    print("=== Avg Claim Counts ===")
    print_table(groups, COUNT_FIELDS, records, 14, 1)
    for f in COUNT_FIELDS:
        print(f"  {f:<16} {METRIC_DEFS.get(f, '')}")


if __name__ == "__main__":
    main()

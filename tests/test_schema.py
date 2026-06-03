"""Schema contract tests for QA pipeline JSONL output files.

Validates that each JSONL output file conforms to the expected schema.
Read-only — does not modify any files or call external APIs.
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

EXPECTED_SCHEMAS: dict[str, set[str]] = {
    "wikipedia_en.jsonl": {"article_id", "id", "url", "title", "text"},
    "qa_pairs.jsonl": {"difficulty", "question", "answer", "title", "source_text_length", "article_id"},
    "qa_pairs_evaluated.jsonl": {"difficulty", "question", "answer", "title", "source_text_length", "article_id", "faithfulness", "completeness", "clarity", "judge_reasoning", "eval_time_s"},
    "qa_pairs_gemma.jsonl": {"difficulty", "question", "answer", "title", "source_text_length", "article_id", "model_answer", "model"},
    "qa_pairs_gemma_eval.jsonl": {"difficulty", "question", "answer", "title", "source_text_length", "article_id", "model_answer", "model", "faithfulness", "completeness", "clarity", "correctness", "judge_reasoning", "eval_time_s"},
}


def check_file(filename: str, expected_keys: set[str]) -> list[str]:
    path = REPO_ROOT / filename
    if not path.exists():
        return [f"SKIP: {filename} not found"]

    errors: list[str] = []
    lines = [l for l in path.read_text().split("\n") if l.strip()]
    if not lines:
        return [f"{filename}: file is empty"]

    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{filename}[{i}]: invalid JSON — {e}")
            continue

        if not isinstance(record, dict):
            errors.append(f"{filename}[{i}]: not a dict")
            continue

        missing = expected_keys - set(record.keys())
        if missing:
            errors.append(f"{filename}[{i}]: missing keys {missing}")

        for key in ("source_text_length", "article_id"):
            if key in record and not isinstance(record[key], int):
                errors.append(f"{filename}[{i}]: {key} is not int")

        for str_key in ("question", "answer", "title", "difficulty"):
            if str_key in record and (not isinstance(record[str_key], str) or len(record[str_key]) == 0):
                errors.append(f"{filename}[{i}]: {str_key} is empty or not a string")

    return errors


def run_checks() -> int:
    all_errors: list[str] = []
    found_any = False

    for filename, expected_keys in EXPECTED_SCHEMAS.items():
        path = REPO_ROOT / filename
        if not path.exists():
            continue
        found_any = True
        errors = check_file(filename, expected_keys)
        if errors:
            all_errors.extend(errors)
            print(f"  {filename}: FAIL ({len(errors)} issue(s))")
            for e in errors:
                print(f"    - {e}")
        else:
            lines = len([l for l in path.read_text().split("\n") if l.strip()])
            print(f"  {filename}: PASS ({lines} records)")

    if not found_any:
        print(f"  (no JSONL files found to validate)")

    if all_errors:
        print(f"\n  FAIL: {len(all_errors)} schema violation(s)")
        return 1
    return 0


def main() -> None:
    t0 = time.monotonic()
    print(f"[schema] validating JSONL output files...")
    rc = run_checks()
    elapsed = time.monotonic() - t0
    status = "PASS" if rc == 0 else "FAIL"
    print(f"[schema] {status} ({elapsed:.1f}s)")
    sys.exit(rc)


if __name__ == "__main__":
    main()

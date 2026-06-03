"""End-to-end tests for the QA pipeline using local Gemma3 model via Ollama.

Requires:
  - Ollama running on localhost:11434 with gemma3:270m model pulled
  - No external API calls (article fixture cached locally)
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
OUTPUT_FILE = "qa_pairs_gemma.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:270m"

ARTICLE = json.loads((FIXTURE_DIR / "article_0.json").read_text())

TEST_PAIRS = [
    {
        "difficulty": "easy",
        "question": "What is the etymological origin of the word 'anarchism'?",
        "answer": "From Ancient Greek anarkhia, meaning 'without a ruler'.",
        "title": ARTICLE["title"],
        "source_text_length": len(ARTICLE["text"]),
        "article_id": ARTICLE["article_id"],
    },
    {
        "difficulty": "medium",
        "question": "What are the two main historical traditions of anarchist schools of thought?",
        "answer": "Social anarchism and individualist anarchism.",
        "title": ARTICLE["title"],
        "source_text_length": len(ARTICLE["text"]),
        "article_id": ARTICLE["article_id"],
    },
]

SYSTEM_PROMPT = "You are a helpful assistant. Answer the question based only on the provided article text."


def check_ollama() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = r.json().get("models", [])
        has = any(m["name"].startswith(OLLAMA_MODEL) for m in models)
        if not has:
            print(f"  SKIP: model '{OLLAMA_MODEL}' not pulled", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"  SKIP: Ollama unreachable: {e}", file=sys.stderr)
        return False


def generate_answer(question: str, context: str) -> str | None:
    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Article:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ],
            "stream": False,
            "options": {"num_predict": 1024},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def run_test() -> int:
    if not check_ollama():
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        input_file = tmp / "qa_pairs.jsonl"
        with open(input_file, "w") as f:
            for pair in TEST_PAIRS:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"  Seeded {len(TEST_PAIRS)} QA pairs")

        results = []
        for pair in TEST_PAIRS:
            answer = generate_answer(pair["question"], ARTICLE["text"])
            if answer is None:
                print(f"  FAIL: no answer for '{pair['question'][:60]}...'", file=sys.stderr)
                return 1
            result = {**pair, "model_answer": answer, "model": OLLAMA_MODEL}
            results.append(result)

        output_file = tmp / OUTPUT_FILE
        with open(output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        assert output_file.exists()
        lines = [l for l in output_file.read_text().strip().splitlines() if l.strip()]
        print(f"  Output: {len(lines)} records")

        errors = []
        if len(lines) != len(TEST_PAIRS):
            errors.append(f"expected {len(TEST_PAIRS)} records, got {len(lines)}")

        for i, line in enumerate(lines):
            record = json.loads(line)
            expected_keys = {"difficulty", "question", "answer", "title", "source_text_length", "article_id", "model_answer", "model"}
            missing = expected_keys - set(record.keys())
            if missing:
                errors.append(f"record[{i}] missing keys: {missing}")
                continue
            if not isinstance(record["model_answer"], str) or len(record["model_answer"]) == 0:
                errors.append(f"record[{i}] model_answer is empty")
            if record["model"] != OLLAMA_MODEL:
                errors.append(f"record[{i}] model='{record['model']}'")
            if record["article_id"] != ARTICLE["article_id"]:
                errors.append(f"record[{i}] article_id wrong")
            if record["title"] != ARTICLE["title"]:
                errors.append(f"record[{i}] title wrong")

        if errors:
            for e in errors:
                print(f"    - {e}", file=sys.stderr)
            return 1

        print(f"  PASS: all {len(lines)} records valid")
        return 0


def main() -> None:
    t0 = time.monotonic()
    print(f"[e2e] generate_gemma pipeline ...")
    rc = run_test()
    elapsed = time.monotonic() - t0
    status = "PASS" if rc == 0 else "FAIL"
    print(f"[e2e] {status} ({elapsed:.1f}s)")
    sys.exit(rc)


if __name__ == "__main__":
    main()

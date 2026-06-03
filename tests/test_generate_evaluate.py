"""Test QA generation + evaluation pipeline using Gemma3 via Ollama.

Requires:
  - Ollama running on localhost:11434 with gemma3:270m model pulled
  - No external API calls (article fixture cached locally)

Exercises the same prompt/logic patterns as generate_qa.py and evaluate_qa.py
but routes through local Gemma instead of DeepSeek.
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:270m"

ARTICLE = json.loads((FIXTURE_DIR / "article_0.json").read_text())

GENERATION_SYSTEM_PROMPT = """You are an expert question-answer pair generator. Given a Wikipedia article, generate exactly 3 question-answer pairs based ONLY on the provided text.

Generate:
- 1 EASY question
- 1 MEDIUM question
- 1 HARD question

Rules:
- Questions must be answerable from the provided text alone
- Answers must be factually grounded in the text
- Vary question types (who, what, when, why, how, compare, explain)

Respond in JSON. Use keys "question" and "answer" for each pair. Tag each with "difficulty" ("easy", "medium", or "hard")."""

GENERATION_USER_TEMPLATE = """Article Title: {title}

Article Text:
{text}

Generate 3 question-answer pairs (1 easy, 1 medium, 1 hard) based on this article. Respond in JSON."""

EVALUATION_SYSTEM_PROMPT = """You are an expert QA evaluator. Given a Wikipedia article, a question, and an answer, rate the answer.

Score 1-5: 1 = completely wrong, 5 = perfectly accurate and complete.

Respond in JSON with keys "score" (int) and "reasoning" (string)."""

EVALUATION_USER_TEMPLATE = """## Wikipedia Article
{text}

## Question
{question}

## Generated Answer
{answer}

Evaluate this answer."""


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


def strip_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[start:end])
    return content


def try_parse_json(content: str) -> dict | list | None:
    content = strip_fences(content)
    for wrapper in ("json", ""):
        text = content.strip()
        if wrapper and text.startswith(wrapper):
            text = text[len(wrapper):].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    # Gemma 270M often emits unescaped quotes inside JSON strings.
    # Try extracting score via regex as last resort.
    score_m = re.search(r'"score":\s*(\d)', content)
    reasoning_m = re.search(r'"reasoning":\s*"(.+?)"(?:\s*[}\]])', content, re.DOTALL)
    if score_m:
        result = {"score": int(score_m.group(1))}
        if reasoning_m:
            result["reasoning"] = reasoning_m.group(1)
        return result
    return None


QA_KEY_ALIASES = {
    "question": ("question", "q", "qa_pair"),
    "answer": ("answer", "a", "response", "model_answer"),
    "difficulty": ("difficulty", "diff", "level"),
}


def normalize_pair(pair: dict) -> dict | None:
    out = {}
    for target, aliases in QA_KEY_ALIASES.items():
        for alias in aliases:
            if alias in pair:
                out[target] = pair[alias]
                break
    if "question" not in out or "answer" not in out:
        return None
    return out


def extract_pairs(parsed: dict | list) -> list[dict]:
    if isinstance(parsed, list):
        candidates = parsed
    elif isinstance(parsed, dict):
        candidates = parsed.get("qa_pairs", parsed.get("pairs", [parsed]))
        if isinstance(candidates, dict):
            candidates = list(candidates.values())
    else:
        return []

    pairs = [normalize_pair(p) for p in candidates if isinstance(p, dict)]
    return [p for p in pairs if p is not None]


def test_generate_qa_pairs() -> int:
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": GENERATION_USER_TEMPLATE.format(title=ARTICLE["title"], text=ARTICLE["text"])},
    ]

    raw = call_ollama(messages, 2048)
    if raw is None:
        print(f"  FAIL: could not parse JSON from Gemma", file=sys.stderr)
        return 1

    pairs = extract_pairs(raw)
    if len(pairs) < 1:
        print(f"  FAIL: no valid QA pairs found (raw keys: {list(raw.keys()) if isinstance(raw, dict) else 'list'})", file=sys.stderr)
        return 1

    errors = []
    for i, pair in enumerate(pairs):
        if len(pair["question"]) < 5:
            errors.append(f"pair[{i}] question too short ({len(pair['question'])} chars)")
        if len(pair["answer"]) < 5:
            errors.append(f"pair[{i}] answer too short ({len(pair['answer'])} chars)")

    if errors:
        for e in errors:
            print(f"    - {e}", file=sys.stderr)
        return 1

    print(f"  Generated {len(pairs)} valid QA pairs")
    for p in pairs:
        diff = p.get("difficulty", "?")
        print(f"    [{diff}] {p['question'][:70]}...")
    return 0


def test_evaluate_qa_pair() -> int:
    question = "What is the etymological origin of the word 'anarchism'?"
    answer = "From Ancient Greek anarkhia, meaning 'without a ruler', composed of an- ('without') and arkhos ('leader')."

    messages = [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": EVALUATION_USER_TEMPLATE.format(text=ARTICLE["text"], question=question, answer=answer)},
    ]

    parsed = call_ollama(messages, 512)
    if parsed is None:
        print(f"  FAIL: could not parse JSON from Gemma", file=sys.stderr)
        return 1

    score = parsed.get("score")
    if score is None:
        score = parsed.get("rating", parsed.get("faithfulness"))
    if not isinstance(score, int):
        print(f"  FAIL: no integer score found (keys: {list(parsed.keys())}, types: {[type(v).__name__ for v in parsed.values()]})", file=sys.stderr)
        return 1
    if score < 1 or score > 5:
        print(f"  FAIL: score={score} out of range", file=sys.stderr)
        return 1

    print(f"  Evaluation: score={score}")
    return 0


def call_ollama(messages: list[dict], max_tokens: int) -> dict | list | None:
    resp = httpx.post(
        OLLAMA_CHAT_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False, "options": {"num_predict": max_tokens}},
        timeout=180,
    )
    resp.raise_for_status()
    return try_parse_json(resp.json()["message"]["content"])


def run_all() -> int:
    if not check_ollama():
        return 0

    tests = [
        ("generate_qa_pairs", test_generate_qa_pairs),
        ("evaluate_qa_pair", test_evaluate_qa_pair),
    ]

    failed = 0
    for name, fn in tests:
        t0 = time.monotonic()
        print(f"  [{name}] ...", end=" ", flush=True)
        rc = fn()
        elapsed = time.monotonic() - t0
        status = "PASS" if rc == 0 else "FAIL"
        print(f" {status} ({elapsed:.1f}s)")
        if rc != 0:
            failed += 1

    return 1 if failed > 0 else 0


def main() -> None:
    t0 = time.monotonic()
    print(f"[generate+evaluate] via {OLLAMA_MODEL}...")
    rc = run_all()
    elapsed = time.monotonic() - t0
    status = "PASS" if rc == 0 else "FAIL"
    print(f"[generate+evaluate] {status} ({elapsed:.1f}s)")
    sys.exit(rc)


if __name__ == "__main__":
    main()

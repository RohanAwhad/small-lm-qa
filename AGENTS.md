# qa — Wikipedia QA Dataset Pipeline

## Structure
- `download_wikipedia.py` — downloads ALL ~6.4M Wikipedia English articles from HF datasets server API to JSONL (resumable, chunked concurrency)
- `utils/wikipedia_loader.py` — shared module: loads articles from local `wikipedia_en.jsonl` by article_id or sequentially
- `generate_qa.py` — generates 15 QA pairs (5 each easy/medium/hard) per article using DeepSeek V4 Flash; loads articles from local JSONL
- `evaluate_ragas.py` — **primary eval**: RAGAS-style claim decomposition → P/R/F1 per pair. Dual mode: reference-based (uses `model_answer`) or article-based (against Wikipedia text). Has resume support.
- `verify_golden.py` — 4-vote LLM judge against Wikipedia article; unanimous = correct. No resume support (clears output on rerun).
- `generate_gemma_answers.py` — re-answers questions using Gemma3 270M via local Ollama; streaming output + tenacity retry
- `summarize_scores.py` — pretty-prints RAGAS eval scores (P/R/F1, claim counts) grouped by difficulty
- `evaluate_qa.py` — **legacy**: 1-5 judge format (faithfulness/completeness/clarity), superseded by `evaluate_ragas.py`
- `compare_gemma.py` — **legacy**: evaluates Gemma3 against DeepSeek golden using old 1-5 format
- `download_wikipedia.py` — downloads ALL ~6.4M Wikipedia English articles from HF datasets server API to JSONL (resumable, chunked concurrency)
- `tests/test_schema.py` — validates JSONL output files against expected schema (read-only, no API)
- `tests/test_generate_evaluate.py` — QA gen + eval via local Gemma (req Ollama)
- `tests/test_e2e.py` — end-to-end Gemma answer pipeline (req Ollama)

## Commands
```bash
# Download all Wikipedia English articles (~6.4M, resumable)
uv run python download_wikipedia.py [N]  # N optional, defaults to ALL

# Generate QA from N articles (resumes — skips already-processed article_ids)
uv run python generate_qa.py N

# Generate Gemma3 answers (streams output line-by-line, tenacity retry on failures)
uv run python generate_gemma_answers.py [input.jsonl] [-o output.jsonl]

# RAGAS eval: claim-based P/R/F1 (has resume — skips already-evaluated pairs)
uv run python evaluate_ragas.py [input.jsonl] --reference qa_pairs.jsonl [-o output.jsonl]

# 4-vote golden verification against Wikipedia (no resume — overwrites output)
uv run python verify_golden.py [input.jsonl] [-o output.jsonl]

# Summarize RAGAS eval scores
uv run python summarize_scores.py [path/to/ragas_eval.jsonl]

# Tests (standalone scripts, not pytest)
uv run python tests/test_schema.py          # validates existing JSONL files
uv run python tests/test_generate_evaluate.py  # req Ollama + gemma3:270m
uv run python tests/test_e2e.py                # req Ollama + gemma3:270m
```

## Pipeline (current)
1. `generate_qa.py N` → `qa_pairs.jsonl` (golden QA pairs)
2. `generate_gemma_answers.py qa_test.json` → `qa_test_gemma.jsonl` (agent answers)
3. `evaluate_ragas.py qa_test_gemma.jsonl --reference qa_pairs.jsonl` → `qa_test_gemma_ragas_eval.jsonl`
4. `summarize_scores.py qa_test_gemma_ragas_eval.jsonl`
5. (optional) `verify_golden.py qa_pairs.jsonl` → `qa_pairs_verified.jsonl`

## RAGAS eval: cached artifacts
- `qa_reference_claims.jsonl` — golden answer claims, keyed by `article_id::question`
- `qa_article_claims.jsonl` — Wikipedia article claims, keyed by `article_id`
- These are regenerated as needed; delete to force re-decomposition

## Dependencies
- `uv` for package management (Python 3.12)
- `DEEPSEEK_API_KEY` env var required for DeepSeek API calls
- `LOGGING_LEVEL` env var controls log verbosity (INFO stderr, DEBUG to `logs/`)
- Ollama tests gracefully skip if `gemma3:270m` not pulled or unreachable

## DeepSeek API quirks
- Model: `deepseek-v4-flash` (thinking enabled by default)
- Does NOT support `json_schema` — only `response_format={"type": "json_object"}`
- Prompt must contain the word `json` when using `json_object`
- Thinking consumes output token budget — don't set `max_tokens` too low
- Do NOT pass `temperature` param (unsupported for this model)
- Thinking tokens available via `choice.message.reasoning_content` but NOT stored by any script

## Code conventions
- All scripts use `asyncio.run(main())` — no sync entrypoints
- Articles loaded from local `wikipedia_en.jsonl` via `wikipedia_loader.py` (pre-downloaded with `download_wikipedia.py`; no longer fetched from HF API on-the-fly)
- `article_id` maps to row index for re-fetching
- `*.jsonl` and `logs/` are gitignored — regenerated artifacts
- Gemma3 270M answers degrade with full article text; needs truncation for large articles
- Pydantic models use `""` default for optional `reasoning` fields (LLM sometimes omits it)

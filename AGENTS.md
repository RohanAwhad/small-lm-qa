# qa — Wikipedia QA Dataset Pipeline

## Structure
- `generate_qa.py` — fetches Wikipedia articles via HF datasets server API, generates 15 QA pairs (5 each easy/medium/hard) per article using DeepSeek V4 Flash
- `evaluate_qa.py` — LLM-as-judge eval of generated QA pairs, scores faithfulness/completeness/clarity 1-5
- `generate_gemma_answers.py` — re-answers questions using Gemma3 270M via local Ollama
- `compare_gemma.py` — evaluates Gemma3 answers against DeepSeek golden, adds correctness metric
- `summarize_scores.py` — pretty-prints eval scores grouped by difficulty
- `tests/test_schema.py` — validates JSONL output files against expected schema (read-only, no API)
- `tests/test_generate_evaluate.py` — QA gen + eval via local Gemma (req Ollama)
- `tests/test_e2e.py` — end-to-end Gemma answer pipeline (req Ollama)

## Commands
```bash
# Generate QA from N articles
uv run python generate_qa.py N

# Evaluate generated QA pairs
uv run python evaluate_qa.py

# Re-answer with Gemma3 (requires Ollama on localhost:11434)
uv run python generate_gemma_answers.py

# Compare Gemma3 answers against DeepSeek golden
uv run python compare_gemma.py

# Summarize eval scores by difficulty
uv run python summarize_scores.py [path/to/evaluated.jsonl]

# Tests (standalone scripts, not pytest)
uv run python tests/test_schema.py          # validates existing JSONL files
uv run python tests/test_generate_evaluate.py  # req Ollama + gemma3:270m
uv run python tests/test_e2e.py                # req Ollama + gemma3:270m
```

## Dependencies
- `uv` for package management (Python 3.12)
- `DEEPSEEK_API_KEY` env var required for DeepSeek API calls
- `LOGGING_LEVEL` env var controls log verbosity (INFO stderr, DEBUG to `logs/`)
- Ollama tests gracefully skip if `gemma3:270m` not pulled or unreachable

## DeepSeek API quirks
- Model: `deepseek-v4-flash` (thinking by default)
- Does NOT support `json_schema` — only `response_format={"type": "json_object"}`
- Prompt must contain the word `json` when using `json_object`
- Thinking consumes output token budget — don't set `max_tokens` too low

## Pipeline notes
- All scripts are async (`asyncio.run(main())`)
- Articles fetched via HF datasets server API (not `datasets` library — too slow)
- `article_id` maps to row index for re-fetching
- `generate_qa.py` has resume support — skips already-processed `article_id`s
- Data flow: generate_qa → evaluate_qa → (optional) generate_gemma_answers → compare_gemma
- `*.jsonl` and `logs/` are gitignored — regenerated artifacts
- Gemma3 270M answers degrade with full article text; needs truncation

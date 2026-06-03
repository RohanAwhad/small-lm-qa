# qa — Wikipedia QA Dataset Pipeline

## Structure

- `generate_qa.py` — fetches Wikipedia articles via HF datasets server API, generates 15 QA pairs (5 each easy/medium/hard) per article using DeepSeek V4 Flash
- `evaluate_qa.py` — LLM-as-judge eval of generated QA pairs, scores faithfulness/completeness/clarity 1-5
- `generate_gemma_answers.py` — takes existing QA pairs, re-answers questions using Gemma3 270M via local Ollama
- `compare_gemma.py` — evaluates Gemma3 answers against DeepSeek golden answers, adds correctness metric

## Commands

```bash
# Generate QA from N Wikipedia articles
uv run python generate_qa.py N

# Evaluate generated QA pairs
uv run python evaluate_qa.py

# Re-answer with Gemma3 (requires Ollama running on localhost:11434)
uv run python generate_gemma_answers.py

# Compare Gemma3 answers against DeepSeek golden
uv run python compare_gemma.py
```

## Dependencies

- `uv` for package management (pyproject.toml + uv.lock)
- `LOGGING_LEVEL` env var controls log verbosity (INFO by default on stderr, DEBUG to logs/ directory)
- `DEEPSEEK_API_KEY` env var required for DeepSeek API calls

## DeepSeek API quirks

- Model: `deepseek-v4-flash` (supports thinking by default)
- Does NOT support OpenAI structured outputs (`json_schema`) — only `json_object`
- Prompt must contain the word `json` when using `response_format={"type": "json_object"}`
- Thinking consumes output token budget — don't set `max_tokens` too low

## Wikipedia data

- Fetched directly from `https://datasets-server.huggingface.co/rows` (not the `datasets` library — too slow)
- `article_id` stored in output JSONL maps to the row index for re-fetching
- Each QA entry includes `title`, `article_id`, `source_text_length`

## Output files

| File | Description |
|---|---|
| `qa_pairs.jsonl` | Generated QA pairs |
| `qa_pairs_evaluated.jsonl` | Eval scores on generated pairs |
| `qa_pairs_gemma.jsonl` | Gemma3 re-answers |
| `qa_pairs_gemma_eval.jsonl` | Eval scores on Gemma3 vs golden |
| `logs/*.log` | Debug logs (gitignored) |

## Other

- No tests, no lint/typecheck config
- Gemma3 answers degrade with full article text vs 4K char truncation (270M model can't focus on noise)
- `*.jsonl` and `logs/` are gitignored — regenerated artifacts

# qa — Wikipedia QA Dataset Pipeline

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
```

## Tests

All tests use plain Python (no pytest). Requires `gemma3:270m` pulled in Ollama for e2e tests.

```bash
uv run python tests/test_schema.py           # validates all JSONL output files (zero deps)
uv run python tests/test_e2e.py              # gemma answer generation pipeline (Ollama)
uv run python tests/test_generate_evaluate.py # QA generation + evaluation prompts (Ollama)
```

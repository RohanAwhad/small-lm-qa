# src/evals — RAGAS-style Claim-based Evaluation

## Architecture

Three-step pipeline: **decompose → classify → score**

```
ref_answer ──→ decompose_answer() ──→ ref_claims
                                                  ╲
                                                   classify_claims() ──→ verdicts ──→ compute_score() ──→ P/R/F1
                                                  ╱
agent_answer ─→ decompose_answer() ──→ agent_claims
```

## Modules

- `decompose.py` — `decompose_answer(client, semaphore, answer, model) -> list[str] | None`
  - Breaks answer text into atomic factual claims via LLM
  - Tenacity retry: 20 attempts, exponential backoff + jitter, max 30s wait

- `classify.py` — `classify_claims(client, semaphore, agent_claims, ref_claims, model) -> ClassifyOutput | None`
  - Classifies each agent claim as SUPPORTED/CONTRADICTED/UNSUPPORTED against ref claims
  - Also reports which ref claims were uncovered (missed by agent)
  - Same tenacity retry config as decompose

- `scoring.py` — `compute_score(ClassifyOutput) -> Score`
  - Pure function, no LLM. Counts verdicts → precision/recall/F1
  - precision = supported / (supported + contradicted + unsupported)
  - recall = supported / (supported + uncovered)

- `run.py` — orchestration
  - `evaluate_single(client, semaphore, ref_answer, agent_answer, model) -> EvalResult` — single pair
  - `evaluate_all(input_path, output_path, ...) -> list[dict]` — batch with CLI

## Data format

- **Reference file** (`qa_pairs_chunked_test.jsonl`): uses `regen_answer` field as ground truth
- **Input file** (agent answers): uses `model_answer` field
- **Pairs matched by**: `article_id` + `question`
- **Output**: JSONL with original fields + ref_claims, agent_claims, verdicts, P/R/F1

## CLI

```bash
# Basic usage
uv run python -m src.evals.run <input.jsonl> -o <output.jsonl>

# All options
uv run python -m src.evals.run <input.jsonl> \
  -o <output.jsonl> \
  --reference qa_pairs_chunked_test.jsonl \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --max-concurrent 50 \
  --overwrite

# Self-hosted DeepSeek on GPU node
uv run python -m src.evals.run <input.jsonl> -o <output.jsonl> \
  --base-url http://localhost:8000/v1 \
  --model deepseek-ai/DeepSeek-V4-Flash
```

## Environment variables

- `DEEPSEEK_API_KEY` — required (use `dummy` for self-hosted)

## Design decisions

- No resume support — if output exists, raises `FileExistsError` (pass `--overwrite` to force)
- No claim caching — ref claims decomposed fresh each run
- `<reasoning>...</reasoning>` tags stripped from agent answers before decomposition
- Both decompose calls (ref + agent) run in parallel per pair via `asyncio.gather`

# src/generation — Answer Generation Pipeline

## Architecture

```
input (QAPair JSONL) → load + validate → build messages → engine (hf or vllm) → write output JSONL
```

## Modules

- `constants.py` — `SamplingParams` frozen dataclass (temperature=0.7, max_new_tokens=8192, do_sample=True), `SYSTEM_PROMPT`, `USER_MESSAGE_TEMPLATE`, `DEFAULT_SAMPLING_PARAMS`

- `utils.py` — `generate_messages(context, question) -> list[dict[str, str]]`

- `hf_engine.py` — `generate(messages_list, model_id, batch_size) -> list[str]`
  - Loads model via AutoModelForCausalLM (bfloat16, SDPA, device_map=auto)
  - Batched inference with per-batch progress logging

- `vllm_engine.py` — `generate(messages_list, model, max_concurrent, base_url) -> list[str]`
  - AsyncOpenAI client with semaphore-gated concurrency
  - Tenacity retry: 20 attempts, exponential backoff + jitter, max 30s wait

- `run.py` — orchestration + CLI
  - `QAPair` pydantic model (16 fields matching `qa_pairs_chunked_test.jsonl`)
  - `run(input_path, output_path, engine, model, batch_size, base_url, overwrite)`
  - Dispatches to `hf_engine.generate()` or `vllm_engine.generate()`

## Input contract

Input must be JSONL matching `qa_pairs_chunked_test.jsonl` schema with `context_chunks` field. Validated into `list[QAPair]` via pydantic.

## Output format

Same as input fields plus `model_answer` and `model`:
```json
{...qa_pair_fields, "model_answer": "...", "model": "..."}
```

## CLI

```bash
# HF (default engine)
uv run python -m src.generation.run <input.jsonl> -o <output.jsonl> -m <model_id>

# vLLM
uv run python -m src.generation.run <input.jsonl> -o <output.jsonl> --engine vllm -m <model> --base-url http://localhost:8000/v1

# On remote GPU node (HF)
.venv/bin/python -m src.generation.run <input.jsonl> -o <output.jsonl> -m model_weights/gemma3-270m/hf_ckpts/checkpoint-500
```

## Environment variables

- `OPENAI_API_KEY` — for vllm engine (default: `dummy` for self-hosted)

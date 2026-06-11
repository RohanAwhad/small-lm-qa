# qa — Wikipedia QA Dataset Pipeline

Fine-tune Gemma3 270M for open-book QA with chain-of-thought reasoning. Three-stage training: SFT → DPO → GKD.

## Pipeline overview

```
Wikipedia articles (download_wikipedia.py)
    ↓
generate_qa.py           → qa_pairs.jsonl           (15 QA pairs per article via DeepSeek)
generate_chunked_qa.py   → qa_pairs_chunked.jsonl   (chunk + BM25 retrieve + regen with reasoning)
split_train_test.py      → qa_train.json / qa_test.json
    ↓
train_hf_gemma3.py       → SFT checkpoint           (full fine-tune on chunked QA)
    ↓
generate_dpo_rollouts.py → dpo_rollouts.jsonl        (5 rollouts per prompt via vLLM)
judge_dpo_rollouts.py    → dpo_rollouts_judged.jsonl (LLM-as-judge picks best/worst)
build_dpo_pairs.py       → dpo_pairs_train.jsonl     (trl DPOTrainer format)
train_dpo_gemma3.py      → DPO checkpoint            (preference alignment)
    ↓
train_gkd_gemma3.py      → GKD checkpoint            (Gemma3 4B teacher distillation)
```

## Data generation

```bash
# Download Wikipedia articles (~6.4M, resumable)
uv run python download_wikipedia.py [N]

# Generate golden QA pairs (resumes)
uv run python generate_qa.py N

# Chunk + retrieve + regenerate with reasoning (resumes)
uv run python generate_chunked_qa.py [N]

# Train/test split
uv run python split_train_test.py
```

## Training

All training runs on remote H100 GPU nodes. Use `.venv/bin/python` on the node (not `uv run`).

```bash
# SFT: full fine-tune on chunked QA
.venv/bin/python train_hf_gemma3.py

# DPO: preference alignment on top of SFT checkpoint
# Single GPU or multi-GPU
.venv/bin/python train_dpo_gemma3.py
torchrun --nproc_per_node=8 train_dpo_gemma3.py

# GKD: on-policy distillation from Gemma3 4B teacher (2 GPUs)
.venv/bin/python train_gkd_gemma3.py
```

### Training configs

| Stage | Model | LR | Batch | Beta | Seq len |
|-------|-------|----|-------|------|---------|
| SFT | `unsloth/gemma-3-270m-it` | 3e-5 | 64 | — | 1024 |
| DPO | SFT checkpoint | 1e-6 | 64 | 5.0 | 1024 |
| GKD | DPO checkpoint (student) + `google/gemma-3-4b-it` (teacher) | 1e-4 | 32 | — | 1024 |

Training format: `system: "Answer the question using the provided context." | user: context+question | assistant: <reasoning>...</reasoning> + answer`

## Evaluation

Answer generation (modular, via `src/generation/`):

```bash
# HF Transformers on GPU (default engine)
uv run python -m src.generation.run qa_pairs_chunked_test.jsonl -o qa_test_hf.jsonl -m <model_or_checkpoint>

# vLLM API (when model is served)
uv run python -m src.generation.run qa_pairs_chunked_test.jsonl -o qa_test_vllm.jsonl \
  --engine vllm -m <model> --base-url http://localhost:8000/v1

# On remote GPU node
.venv/bin/python -m src.generation.run qa_pairs_chunked_test.jsonl -o qa_test_hf.jsonl -m model_weights/gemma3-270m/hf_ckpts/checkpoint-500
```

RAGAS evaluation (claim-based P/R/F1):

```bash
# New modular eval (src/evals/)
uv run python -m src.evals.run qa_test_gemma.jsonl -o ragas_eval.jsonl
uv run python -m src.evals.run qa_test_gemma.jsonl -o ragas_eval.jsonl --overwrite  # if output exists

# With self-hosted DeepSeek judge
uv run python -m src.evals.run qa_test_gemma.jsonl -o ragas_eval.jsonl \
  --base-url http://localhost:8000/v1 --model deepseek-ai/DeepSeek-V4-Flash

# Summarize scores
uv run python summarize_scores.py ragas_eval.jsonl

# Legacy eval (evaluate_ragas.py) — still works but superseded by src/evals/
uv run python evaluate_ragas.py qa_test_gemma.jsonl --reference qa_pairs.jsonl -o ragas_eval.jsonl
```

## Tests

Standalone scripts (no pytest). Ollama tests need `gemma3:270m` pulled.

```bash
uv run python tests/test_schema.py
uv run python tests/test_generate_evaluate.py
uv run python tests/test_e2e.py
```

## Dependencies

- `uv` (Python 3.12)
- `DEEPSEEK_API_KEY` for QA generation and evaluation
- `WANDB_API_KEY` for training logging (project: `small-lm-qa`)
- H100 GPU for training

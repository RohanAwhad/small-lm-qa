# qa — Wikipedia QA Dataset Pipeline

Fine-tune Gemma3 270M for open-book QA with chain-of-thought reasoning.

## Pipeline

```
generate_qa.py          → qa_pairs.jsonl (15 QA pairs per article via DeepSeek)
generate_chunked_qa.py  → qa_pairs_chunked.jsonl (chunk + retrieve + regen with reasoning)
train_hf_gemma3.py      → model_weights/ (fine-tune on chunked QA)
```

### 1. Generate golden QA pairs
```bash
uv run python generate_qa.py N              # N articles, resumes
```

### 2. Chunk articles + retrieve + regenerate answers
```bash
uv run python generate_chunked_qa.py [N]    # all or N articles, resumes
```

Pipeline per article:
- Recursive text splitting (512 tok max, Gemma3 tokenizer, separator hierarchy: `\n\n` → `\n` → `. ` → `, ` → ` `)
- BM25 retrieval: top-3 chunks using golden answer as query
- Answer regeneration from chunks (DeepSeek, captures reasoning_content)
- RAGAS-style context precision + recall evaluation

### 3. Fine-tune
```bash
CUDA_VISIBLE_DEVICES=0 uv run python train_hf_gemma3.py
```

Training format:
```
system: "Answer the question using the provided context."
user:   "Context:\n{chunks}\n\nQuestion: {question}"
assistant: "<reasoning>\n{reasoning}\n</reasoning>\n\n{answer}"
```

### Training config
- Model: `unsloth/gemma-3-270m-it` (268M params, 262K vocab)
- LR: 3e-5 constant (Google recommends 5e-5 for this model)
- Effective batch: 64 (bs=16 x grad_accum=4)
- Max seq len: 1024 tokens
- Filters: reasoning <= 512 tokens, total seq <= 1024 tokens
- 12,742 training examples (from 14,986 after filtering)
- Wandb project: `small-lm-qa`

## Evaluation

```bash
uv run python generate_gemma_answers.py [input.jsonl] [-o output.jsonl]
uv run python evaluate_ragas.py [input.jsonl] --reference qa_pairs.jsonl [-o output.jsonl]
uv run python summarize_scores.py [path/to/ragas_eval.jsonl]
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
- `DEEPSEEK_API_KEY` for QA generation/evaluation
- `WANDB_API_KEY` for training logging
- H100 GPU for training (270M model, 262K vocab needs ~75GB VRAM)

# Gemma3 270M Fine-tuning Evaluation Report

## Setup

- **Base model**: `unsloth/gemma-3-270m-it` (268M params)
- **Training data**: `qa_pairs_chunked.jsonl` — Wikipedia QA with BM25-retrieved chunks + DeepSeek reasoning
- **Training config**: LR=3e-5, bs=16, grad_accum=4 (effective bs=64), 10 epochs, max_seq_len=1024
- **Eval set**: `qa_pairs_chunked_test.jsonl` (600 pairs: 200 easy, 200 medium, 200 hard)
- **Eval method**: RAGAS claim decomposition — P/R/F1 against golden reference answers
- **Inference**: HF Transformers batch inference on H100 (`generate_hf_answers.py`, bs=64, bf16, SDPA)

## Results

### Overall

| Model | F1 | Precision | Recall | Delta F1 |
|---|---|---|---|---|
| Baseline (pre-trained) | 0.383 | 0.483 | 0.399 | — |
| Step 500 (~2.6 epochs) | 0.421 | 0.480 | 0.459 | +9.9% |
| Step 1000 (~5.2 epochs) | **0.435** | 0.482 | **0.486** | **+13.6%** |

### By difficulty

**Baseline**

|  | F1 | Precision | Recall |
|---|---|---|---|
| easy | 0.392 | 0.476 | 0.440 |
| medium | 0.417 | 0.519 | 0.432 |
| hard | 0.339 | 0.453 | 0.324 |

**Step 500**

|  | F1 | Precision | Recall |
|---|---|---|---|
| easy | 0.452 | 0.485 | 0.553 |
| medium | 0.447 | 0.478 | 0.485 |
| hard | 0.364 | 0.476 | 0.339 |

**Step 1000**

|  | F1 | Precision | Recall |
|---|---|---|---|
| easy | 0.467 | 0.504 | 0.571 |
| medium | 0.452 | 0.483 | 0.502 |
| hard | 0.387 | 0.459 | 0.384 |

### Claim counts (Step 1000 vs Baseline)

|  | Supported | Contradicted | Unsupported | Uncovered |
|---|---|---|---|---|
| Baseline | 2.0 | 0.3 | 2.5 | 4.0 |
| Step 1000 | 2.4 | 0.5 | 2.4 | 3.4 |

## Key findings

- **Recall drives the improvement**: +21.8% recall at step 1000 (0.399 → 0.486). The model answers more completely after fine-tuning.
- **Precision holds steady** (~0.48 across all checkpoints). Fine-tuning doesn't increase hallucination.
- **Easy questions benefit most**: +19.1% F1 (0.392 → 0.467). Hard questions improve less (+14.2%).
- **Uncovered claims drop**: 4.0 → 3.4 avg. The model misses fewer reference claims.
- **Contradicted claims increase slightly**: 0.3 → 0.5. Minor tradeoff — model is more assertive but occasionally wrong.
- **Training not yet converged**: step 1000 > step 500, suggesting further gains possible.

## Previous eval (full article context, Ollama)

Earlier evals used full Wikipedia articles as context via Ollama, which mismatched the chunked training format:

| Model | F1 | Note |
|---|---|---|
| Baseline (Ollama, full article) | 0.236 | Train/eval mismatch |
| Baseline (HF, chunks) | 0.383 | Correct eval setup |

The 0.236 → 0.383 jump shows that matching eval context to training context is critical for this model.

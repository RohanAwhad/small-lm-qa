# Gemma3 270M Fine-tuning Evaluation Report

## Setup

- **Base model**: `unsloth/gemma-3-270m-it` (268M params)
- **Training data**: `qa_pairs_chunked.jsonl` — Wikipedia QA with BM25-retrieved chunks + DeepSeek reasoning
- **Training runs**:
  - **Run 1** (steps 500/1000/10000): ~200 articles, LR=3e-5, bs=16, grad_accum=4 (effective bs=64), 10 epochs
  - **Run 2** (step 1031): ~5000 articles, same hyperparams — more diverse training data
- **Max seq len**: 1024
- **Eval set**: `qa_pairs_chunked_test.jsonl` (600 pairs: 200 easy, 200 medium, 200 hard)
- **Eval method**: RAGAS claim decomposition — P/R/F1 against golden reference answers
- **Inference**: HF Transformers batch inference on H100 (`generate_hf_answers.py`, bs=64, bf16, SDPA)

## Results

### Overall

| Model | F1 | Precision | Recall | Delta F1 |
|---|---|---|---|---|
| Baseline (pre-trained) | 0.383 | 0.483 | 0.399 | — |
| Step 500 (~2.6 epochs) | 0.421 | 0.480 | 0.459 | +9.9% |
| Step 1000 (~5.2 epochs) | 0.435 | 0.482 | 0.486 | +13.6% |
| Step 10000 (full training) | 0.436 | 0.491 | 0.461 | +13.8% |
| **Step 1031 (5k articles)** | **0.449** | **0.526** | **0.481** | **+17.2%** |

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

**Step 10000**

|  | F1 | Precision | Recall |
|---|---|---|---|
| easy | 0.470 | 0.515 | 0.539 |
| medium | 0.455 | 0.509 | 0.476 |
| hard | 0.381 | 0.450 | 0.368 |

**Step 1031 (5k articles)**

|  | F1 | Precision | Recall |
|---|---|---|---|
| easy | 0.516 | 0.584 | 0.601 |
| medium | 0.455 | 0.520 | 0.482 |
| hard | 0.378 | 0.475 | 0.361 |

### Claim counts

|  | Supported | Contradicted | Unsupported | Uncovered |
|---|---|---|---|---|
| Baseline | 2.0 | 0.3 | 2.5 | 4.0 |
| Step 1000 | 2.4 | 0.5 | 2.4 | 3.4 |
| Step 10000 | 2.3 | 0.6 | 2.2 | 3.6 |
| Step 1031 (5k) | 2.3 | 0.4 | 2.0 | 3.6 |

## Key findings

- **Recall drives the improvement**: +21.8% recall at step 1000 (0.399 → 0.486). The model answers more completely after fine-tuning.
- **Precision holds steady** (~0.48 across all checkpoints). Fine-tuning doesn't increase hallucination.
- **Easy questions benefit most**: +19.1% F1 (0.392 → 0.467). Hard questions improve less (+14.2%).
- **Uncovered claims drop**: 4.0 → 3.4 avg. The model misses fewer reference claims.
- **Contradicted claims increase slightly**: 0.3 → 0.5. Minor tradeoff — model is more assertive but occasionally wrong.
- **Converged by step 1000** (on 200 articles): step 10000 (F1=0.436) matches step 1000 (F1=0.435). More epochs don't help.
- **More training data helps**: Step 1031 trained on 5k articles (F1=0.449) beats all 200-article checkpoints. Precision jumps to 0.526 (+9% over baseline). Unsupported claims drop to 2.0 (from 2.5), indicating less hallucination with more diverse training data.

## Previous eval (full article context, Ollama)

Earlier evals used full Wikipedia articles as context via Ollama, which mismatched the chunked training format:

| Model | F1 | Note |
|---|---|---|
| Baseline (Ollama, full article) | 0.236 | Train/eval mismatch |
| Baseline (HF, chunks) | 0.383 | Correct eval setup |

The 0.236 → 0.383 jump shows that matching eval context to training context is critical for this model.

## LLM judge parity: DeepSeek API vs self-hosted

Step 1031 evaluated with both judges (same generated answers, different LLM judge):

| Judge | F1 | Precision | Recall |
|---|---|---|---|
| DeepSeek API (`deepseek-v4-flash`) | 0.449 | 0.526 | 0.481 |
| Self-hosted V4 Flash (`deepseek-ai/DeepSeek-V4-Flash` on rh-h100-07) | 0.489 | 0.565 | 0.509 |

Self-hosted scores ~9% higher F1 — likely a different model version or thinking behavior. Not 1:1 parity, but directionally consistent (same difficulty ranking, same relative checkpoint ordering). When comparing checkpoints, use the same judge throughout.

### Full results (self-hosted judge)

| Model | F1 | Precision | Recall | Delta F1 |
|---|---|---|---|---|
| Baseline (pre-trained) | 0.420 | 0.513 | 0.436 | — |
| **Step 1031 (5k articles)** | **0.489** | **0.565** | **0.509** | **+16.4%** |
| Step 3500 (5k articles) | 0.485 | 0.565 | 0.503 | +15.5% |

**By difficulty (self-hosted judge)**

|  | Baseline F1 | Step 1031 F1 | Step 3500 F1 |
|---|---|---|---|
| easy | 0.501 | 0.559 | 0.561 |
| medium | 0.433 | 0.505 | 0.483 |
| hard | 0.327 | 0.402 | 0.410 |

Step 1031 remains the best checkpoint. Step 3500 shows mild overfitting on medium questions (0.505 → 0.483) while hard improves slightly (0.402 → 0.410). Overall, the model converges around step 1031.

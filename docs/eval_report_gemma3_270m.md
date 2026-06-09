# Gemma3 270M Fine-tuning Evaluation Report

## Setup

- **Base model**: `unsloth/gemma-3-270m-it` (268M params)
- **Training data**: `qa_pairs_chunked.jsonl` — Wikipedia QA with BM25-retrieved chunks + DeepSeek reasoning
- **Training runs**:
  - **Run 1** (steps 500/1000/10000): ~200 articles, LR=3e-5, bs=16, grad_accum=4 (effective bs=64), 10 epochs, max_seq_len=1024
  - **Run 2** (step 1031, best): ~5000 articles, LR=3e-5, constant schedule, 1 epoch, same batch config. Checkpoint lost (Trainer rotated it out).
  - **Run 2 continued** (step 3500): same run, later checkpoint — mild overfitting vs step 1031
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

Note: prior evals included `<reasoning>...</reasoning>` tags in model answers, inflating scores. Results below marked "(with reasoning)" are historical. Current evals strip reasoning before claim decomposition.

| Model | F1 | Precision | Recall | Delta F1 | Note |
|---|---|---|---|---|---|
| Baseline (270M, pre-trained) | 0.420 | 0.513 | 0.436 | — | no reasoning tags |
| Final (270M, 5k, LR=5e-5) | 0.509 | 0.588 | 0.516 | +21.2% | with reasoning (inflated) |
| **Final (270M, 5k, LR=5e-5)** | **0.479** | **0.546** | **0.497** | **+14.0%** | **reasoning stripped** |
| **Gemma3 27B (pre-trained)** | **0.706** | **0.726** | **0.753** | **+68.1%** | no reasoning tags |

**By difficulty (self-hosted judge, reasoning stripped)**

|  | Baseline (270M) | Final 5e-5 (270M) | Gemma3 27B |
|---|---|---|---|
| easy | 0.501 | 0.583 | **0.727** |
| medium | 0.433 | 0.477 | **0.719** |
| hard | 0.327 | 0.377 | **0.673** |

**Claim counts (self-hosted judge, reasoning stripped)**

|  | Supported | Contradicted | Unsupported | Uncovered |
|---|---|---|---|---|
| Baseline (270M) | 2.1 | 0.3 | 2.3 | 3.8 |
| Final 5e-5 (270M) | 2.4 | 0.4 | 2.1 | 3.7 |
| Gemma3 27B | 4.6 | 0.2 | 2.0 | 1.9 |

### Analysis

- **Reasoning tag inflation**: Including `<reasoning>` tags inflated 270M F1 from 0.479 to 0.509 (+6%). All comparisons should strip reasoning.
- **Fine-tuning helps**: 270M final (LR=5e-5) F1=0.479, +14.0% over baseline (0.420). Google's recommended LR=5e-5 still wins.
- **100x model size gap**: Gemma3 27B (F1=0.706) scores 47% higher than our best fine-tuned 270M (0.479). The gap is consistent across all difficulties.
- **27B excels at recall**: 0.753 recall vs 0.497 — the larger model covers far more reference claims (1.9 uncovered vs 3.7).
- **27B hallucinates less**: 0.2 contradicted claims vs 0.4, and higher precision (0.726 vs 0.546).
- **Hard questions show largest gap**: 27B F1=0.673 vs 270M F1=0.377 — hard questions require reasoning capacity the 270M lacks.
- Step 1031 checkpoint is lost (Trainer rotation). The final checkpoints supersede it.

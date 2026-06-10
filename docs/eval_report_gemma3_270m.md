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
| SFT Final (270M, 5k, LR=5e-5) | 0.479 | 0.546 | 0.497 | +14.0% | reasoning stripped |
| DPO 1-GPU final | 0.517 | 0.578 | 0.529 | +23.1% | reasoning stripped |
| **DPO 4-GPU step 955 (epoch 1)** | **0.532** | **0.546** | **0.612** | **+26.7%** | **reasoning stripped, best 270M** |
| DPO 4-GPU step 1910 (epoch 2) | 0.507 | 0.524 | 0.581 | +20.7% | reasoning stripped, overfitting |
| **Gemma3 27B (pre-trained)** | **0.706** | **0.726** | **0.753** | **+68.1%** | no reasoning tags |

**By difficulty (self-hosted judge, reasoning stripped)**

|  | Baseline (270M) | SFT 5e-5 | DPO 4-GPU ep1 | DPO 4-GPU ep2 | Gemma3 27B |
|---|---|---|---|---|---|
| easy | 0.501 | 0.583 | 0.558 | 0.548 | **0.727** |
| medium | 0.433 | 0.477 | 0.559 | 0.537 | **0.719** |
| hard | 0.327 | 0.377 | 0.470 | 0.435 | **0.673** |

**Claim counts (self-hosted judge, reasoning stripped)**

|  | Supported | Contradicted | Unsupported | Uncovered |
|---|---|---|---|---|
| Baseline (270M) | 2.1 | 0.3 | 2.3 | 3.8 |
| SFT 5e-5 (270M) | 2.4 | 0.4 | 2.1 | 3.7 |
| DPO 4-GPU ep1 | 2.9 | 0.4 | 2.5 | 2.6 |
| DPO 4-GPU ep2 | 3.0 | 0.5 | 3.0 | 3.2 |
| Gemma3 27B | 4.6 | 0.2 | 2.0 | 1.9 |

### Analysis

- **DPO is the best 270M result**: DPO 4-GPU epoch 1 (F1=0.532) beats SFT (0.479) by +11%. Biggest gain is recall (0.612 vs 0.497).
- **DPO overfits by epoch 2**: F1 drops from 0.532 to 0.507. Unsupported claims rise (2.5 → 3.0), contradicted claims rise (0.4 → 0.5). Train for 1 epoch only.
- **DPO closes the medium/hard gap**: Medium F1 jumps from 0.477 (SFT) to 0.559 (DPO). Hard from 0.377 to 0.470. DPO helps most where SFT struggled.
- **Reasoning tag inflation**: Including `<reasoning>` tags inflated SFT F1 from 0.479 to 0.509 (+6%). All comparisons strip reasoning.
- **100x model size gap remains**: Gemma3 27B (F1=0.706) still 33% higher than best 270M (0.532). Gap is largest on easy questions (0.727 vs 0.558).
- **27B excels at precision and recall**: 0.726 precision and 0.753 recall vs 0.546/0.612 for best 270M. Fewer hallucinations and more complete answers.
- Step 1031 checkpoint is lost (Trainer rotation). The final checkpoints supersede it.

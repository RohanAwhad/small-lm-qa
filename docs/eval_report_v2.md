# Evaluation Report v2

## Changes from v1

- **Eval pipeline**: `src.evals.run` replaces legacy `evaluate_ragas.py`
  - Reference claims decomposed from `regen_answer` (chunked-context regenerated answer), not `answer` (golden answer)
  - Question included in decomposition prompt — fixes terse answers decomposing to useless claims
  - Compound claim handling in classify prompt — agent claims covering multiple ref facts are SUPPORTED
  - Thinking enabled for judge (`chat_template_kwargs: enable_thinking`)
  - No claim caching — fresh decomposition each run
  - Tenacity retry on all LLM calls (20 attempts, exponential backoff)
- **Generation pipeline**: `src.generation.run` replaces legacy `generate_hf_answers.py` / `generate_vllm_answers.py`
  - `max_new_tokens=8192` (was 1024), `temperature=0.7`, `do_sample=True`
  - Shared `SamplingParams` across HF and vLLM engines

## Setup

- **Eval set**: `qa_pairs_chunked_test.jsonl` (600 pairs: 200 easy, 200 medium, 200 hard)
- **Reference**: `qa_pairs_chunked_test.jsonl` `regen_answer` field
- **Judge**: Self-hosted DeepSeek V4 Flash (`deepseek-ai/DeepSeek-V4-Flash` on rh-h100-11, thinking enabled)
- **Max concurrent**: 128

## Results

### Leaderboard

| Model | Params | F1 | Precision | Recall |
|---|---|---|---|---|
| Gemma3 270M baseline | 0.27B | 0.400 | 0.511 | 0.398 |
| Gemma3 270M DPO ep1 | 0.27B | 0.495 | 0.497 | 0.578 |
| Gemma3 270M SFT (5e-5) | 0.27B | 0.505 | 0.572 | 0.510 |
| Gemma3 270M GKD step 1000 | 0.27B | 0.547 | 0.596 | 0.562 |
| DeepSeek V4 Flash | MoE | 0.793 | 0.871 | 0.766 |

### Key findings

- **GKD is the best 270M method** (F1=0.547), ahead of SFT (0.505) and DPO (0.495)
- **DPO dropped below SFT** in v2 — high recall (0.578) but lowest precision (0.497) of all fine-tuned models. DPO generates more claims but many are unsupported.
- **GKD has the best precision** among 270M models (0.596) — distillation from 4B teacher produces more accurate claims
- **DeepSeek ceiling at ~0.79** — 11/600 pairs score F1=0, mostly due to conservative `regen_answer` ("context doesn't specify") or genuinely wrong agent answers

## Remaining classifier failure modes (11 pairs F1=0)

1. **Ref says "context doesn't specify"** (3 pairs) — regen_answer is conservative about dates/details not explicitly in chunks. Agent answers correctly from context. Data quality issue.
2. **Agent adds detail beyond vague ref** (2 pairs) — ref only names categories, agent describes them. Ref too terse.
3. **Classifier too strict on rephrasing** (3 pairs) — agent says same fact in different words, classifier marks UNSUPPORTED despite semantic match.
4. **Agent genuinely wrong** (3 pairs) — correctly scored as CONTRADICTED/UNSUPPORTED.

## v1 vs v2 comparison

| Model | v1 F1 | v2 F1 | Delta |
|---|---|---|---|
| 270M baseline | 0.420 | 0.400 | -0.020 |
| 270M SFT | 0.479 | 0.505 | +0.026 |
| 270M DPO ep1 | 0.532 | 0.495 | -0.037 |
| 270M GKD step 1000 | 0.537 | 0.547 | +0.010 |
| DeepSeek V4 Flash | 0.701 | 0.793 | +0.092 |

The v2 eval is stricter overall (thinking-enabled judge, fresh claim decomposition). DPO's v1 score was inflated — the old eval was more lenient on unsupported claims. GKD and SFT scores are stable or improved. DeepSeek benefits most from the `regen_answer` fix.

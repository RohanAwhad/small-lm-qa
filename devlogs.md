# Devlogs

## 2026-06-09: DPO research and LLM-as-judge prototype

### DPO background research
- Researched Nemotron 3 Nano and Liquid Foundation Models (LFM2) for DPO training details
- Nemotron: ~50K pairs (found 10K sufficient), beta=0.05, LR=3e-6, but DPO was NOT used in final model (RL was sufficient)
- LFM2: ~700K pairs for all sizes including 350M, beta=5.0, LR=8e-7, 2 epochs. Used DPO+APO-zero hybrid with length normalization
- LFM2 most relevant — applied DPO to 350M model (close to our 270M)
- Both used on-policy sampling (generate from the SFT checkpoint being trained)

### DPO pipeline design
- Plan: reuse existing 75K SFT questions from `qa_pairs_chunked_train.jsonl`
- Generate 5 rollouts per question at temp=1.0 from SFT checkpoint
- Serve model via vLLM with data-parallel 8 on rh-h100-01 for throughput
- 75K x 5 = 375K total generations — should be fast for a 270M model

### LLM-as-judge approach
- Instead of RAGAS F1 scoring per rollout, use DeepSeek Flash as a comparative judge
- Judge sees ALL rollouts at once and picks best_idx / worst_idx
- Judge evaluates: reasoning hallucination, reasoning→answer coherence, factual correctness, completeness
- Reasoning tags (`<reasoning>...</reasoning>`) are passed to judge — trains model to reason well, not just answer well
- If all rollouts are bad (`all_bad=true`), skip the question (no DPO signal)

### Prototype results (5 questions, 5 rollouts each)
- 4/5 questions produced usable DPO pairs
- 1/5 skipped (Lakemba — all rollouts gave wrong year, model consistently fails on this)
- Judge correctly identifies hallucinations, incomplete responses, factual errors
- Judge explanations are specific and actionable
- temp=1.0 gives good diversity — clear quality separation between rollouts

### Scripts
- `judge_dpo_rollouts.py` — LLM-as-judge for DPO rollout selection (async, Pydantic-validated, retry)
- Input: JSONL with `{question, golden_answer, context, rollouts: [str]}`
- Output: same + `judgment: {best_idx, worst_idx, all_bad, all_good, explanation}`

### Next steps
- Write `generate_dpo_rollouts.py` — vLLM client to sample N rollouts per prompt
- Full pipeline: generate rollouts → judge → filter → format for DPO training
- DPO training script (likely via `trl` DPOTrainer)
- Decide on beta and other hyperparams (LFM2's beta=5.0 worth trying at 270M scale)

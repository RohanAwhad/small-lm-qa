# On-Policy Knowledge Distillation for Gemma 3 270M

> Generated: 2025-06-10 | Sources: 12 papers, 8 implementations, 15+ community resources

## TL;DR

- **GKD** (Google DeepMind, ICLR 2024 Oral) is the framework: lambda=1.0 (fully on-policy) consistently outperforms all other settings. Student generates, teacher provides soft logits, no gradient through sampling.
- **TRL removed GKDTrainer** from v1.5.1 — must implement ourselves (~300 lines) or use `trl==0.12.2`.
- **115x capacity gap** (31B/270M) is extreme. Standard KD degrades at gaps >2.5x. **TAID** (Sakana AI, ICLR 2025) is the only method with monotonic improvement for large gaps.
- **NF4 teacher fits on 1 GPU**: 31B NF4 = ~16GB + 270M student = ~25GB total. But 262K vocab logits are the real memory bottleneck (~2GB per example).
- **Post-DPO constraint**: Must use KL-divergence loss, NOT cross-entropy, to preserve alignment (NVIDIA QAD finding).

## The GKD Algorithm

```python
# Pseudocode (Agarwal et al., ICLR 2024)
for batch in dataloader:
    prompts = batch["prompt"]
    
    # 1. Student generates (on-policy, lambda=1.0)
    with torch.no_grad():
        student_outputs = student.generate(prompts, temperature=1.0)
    
    # 2. Teacher scores student-generated sequences
    with torch.no_grad():
        teacher_logits = teacher(prompts + student_outputs).logits
    
    # 3. Student re-scores same sequences (WITH gradient)
    student_logits = student(prompts + student_outputs).logits
    
    # 4. Loss on completion tokens only
    loss = generalized_jsd(student_logits, teacher_logits, beta=0.5)
    
    # 5. Update student (no gradient through step 1)
    loss.backward()
    optimizer.step()
```

Key: **two forward passes through student** per step — generation (no_grad, autoregressive) + training (with grad, teacher-forced).

## Loss Functions

| Loss | Formula | When to use | Source |
|------|---------|-------------|--------|
| Forward KL | KL(teacher \|\| student) | Reasoning/QA tasks | GKD (GSM8K) |
| Reverse KL | KL(student \|\| teacher) | Instruction following, mode-seeking | MiniLLM, GKD (FLAN) |
| JSD(beta) | beta*KL(P\|\|M) + (1-beta)*KL(Q\|\|M) | General purpose (beta=0.5 default) | GKD |
| SRKL(0.1) | Skew reverse KL | Best for sub-1B students | DistiLLM (ICML 2024) |
| TAID | Adaptive interpolated KL | Large capacity gaps | TAID (ICLR 2025) |

**Recommendation**: Start with forward KL (simplest, works for QA/reasoning). If student degrades, try TAID.

## The Capacity Gap Problem

**Law of Capacity Gap** (Zhang & Song 2024): optimal teacher = 2.5x student size.

| Student | Optimal Teacher | Our Teacher | Gap |
|---------|----------------|-------------|-----|
| 270M | ~675M | 31B | 115x (extreme) |

**Mitigations:**
1. **TAID** — interpolates from student distribution toward teacher over training. Only method with monotonic improvement for large gaps. Code: github.com/SakanaAI/TAID
2. **Multi-stage**: 31B → 3B → 270M (each step ~10x)
3. **Sequence-level KD**: Just SFT on teacher outputs — bypasses logit matching entirely

## Vocab Mismatch: Cross-Tokenizer Distillation

Same-vocab (Gemma 4 → Gemma 3) avoids this entirely. But for cross-family distillation (e.g., DeepSeek teacher → Gemma student), token-level KL is undefined — P_teacher and P_student are distributions over different event spaces.

### Approach 1: Sequence-Level KL (sidestep the problem)

- Student generates sequence `y` from prompt `x`
- Each model tokenizes `y` with its own tokenizer, computes `log P(y|x)` as sum of token log-probs
- Sequence-level reverse KL: `log P_student(y|x) - log P_teacher(y|x)`
- Use as REINFORCE loss or reward signal
- **Pro:** clean, no alignment needed
- **Con:** high variance (one scalar per sequence), needs variance reduction baselines

### Approach 2: DPO / Ranking (already in pipeline)

- Student generates N rollouts (on-policy)
- Teacher ranks best/worst via LLM-as-judge
- DPO loss — vocab never needs to align
- **This is effectively on-policy distillation with vocab-agnostic ranking**
- Already implemented: `generate_dpo_rollouts.py` → `judge_dpo_rollouts.py` → `build_dpo_pairs.py`
- **Pro:** works today, no new infrastructure
- **Con:** loses fine-grained token-level distributional info from teacher

### Approach 3: Span-Aligned Token-Level KL (most promising for cross-vocab)

Both tokenizers segment the same text differently. Find alignment points and compute KL over aligned spans:

```
Text:      "The quantum computer solved it"
Teacher:   ["The", " quantum", " computer", " solved", " it"]
Student:   ["The", " qu", "antum", " computer", " sol", "ved", " it"]
Aligned:   |"The"| " quantum"      |" computer"| " solved"  |" it"|
           |  T1 |   T2            |    T3     |    T4      | T5  |
           |  S1 | S2      S3     |    S4     | S5     S6  | S7  |
```

1. Tokenize text with both tokenizers
2. Find **character boundary alignment points** where both tokenizers have a split
3. Between two aligned boundaries = a "span"
4. `P_teacher(span) = product of teacher token probs within that span`
5. `P_student(span) = product of student token probs within that span`
6. Compute KL over aligned spans: `sum_spans KL(P_student(span) || P_teacher(span))`

- **Pro:** token-level signal without requiring same vocab, more signal than sequence-level
- **Con:** implementation complexity — need to build the character-boundary alignment, handle edge cases where spans are very uneven
- Spans that cover many tokens in one model but few in another will have noisier gradients

### Approach 4: Byte-Level Marginalization (theoretically cleanest)

Both BPE vocabs decompose to byte sequences. Compute KL at the byte level:

- At each byte position, marginalize over all tokens containing that byte
- Requires prefix-tree traversal over full vocab at each step
- **Pro:** principled, no information loss
- **Con:** computationally expensive (262K vocab prefix-tree marginalization per byte), research-project-level effort

### Recommendation

| Approach | Effort | Signal Quality | Status |
|----------|--------|---------------|--------|
| DPO ranking (approach 2) | None (done) | Coarse (best/worst only) | Already in pipeline |
| Span-aligned KL (approach 3) | Medium (~200 lines) | Good (span-level distributions) | Next to try |
| Sequence-level KL (approach 1) | Low (~100 lines) | Noisy (one scalar/seq) | Fallback |
| Byte-level (approach 4) | High (research) | Best (full distributions) | Skip for now |

For same-vocab teacher (Gemma 4 31B): use standard token-level GKD (no mismatch).
For cross-vocab teacher (DeepSeek): approach 2 is working today, approach 3 is the next step up.

## Memory Analysis (1x H100 80GB)

| Component | Memory |
|-----------|--------|
| Gemma 4 31B NF4 teacher | ~16-18 GB |
| Gemma 3 270M student (weights + optimizer + grads) | ~5-7 GB |
| 262K logits [B=2, L=1024] × 2 models, fp32 | ~4 GB |
| CUDA overhead + KV cache | ~3-5 GB |
| **Total** | **~30-35 GB** |

Fits on 1 H100 with ~45GB spare. Keep batch small (1-2) with grad accumulation.

**262K vocab logit bottleneck**: Full logits = 1MB/token. Mitigations:
- Top-K logits (K=128) — ~2500x storage reduction, minimal quality loss
- Chunked lm_head computation (TRL's `patch_chunked_lm_head`)
- BF16 logits + FP32 loss computation
- Liger kernel for fused JSD (avoids materializing full logit tensor)

## Implementation Options

| Option | Pros | Cons |
|--------|------|------|
| **Custom training loop** (~300 lines) | Full control, no deps | Must handle DDP, generation unwrapping |
| **Vendor trl v0.12.2 GKDTrainer** | Battle-tested, handles edge cases | Removed for a reason, may have bugs |
| **trl.experimental.GKDTrainer** (if available) | Official | Check if it exists in current trl |
| **veRL framework** | Industrial-grade, used by DeepSeek/Qwen | Heavier dependency, overkill for 270M |

## Recommended Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Lambda (on-policy fraction) | **1.0** | GKD — pure on-policy always best |
| Beta (JSD interpolation) | **0.0** (forward KL) or **0.5** (JSD) | GKD — forward KL for QA/reasoning |
| Temperature (generation) | **1.0** | LLM distributions are already soft |
| Temperature (teacher logits) | **1.0** (try 0.1) | GKD — T=0.1 helps for reasoning |
| Learning rate | **1e-4 to 3e-4** | torchtune, GKD |
| Optimizer | AdamW, eps=1e-6 | bf16 stability |
| Batch size | **1-2** with grad_accum=16-32 | 262K vocab memory constraint |
| Training steps | **200-500** on-policy steps | On-policy is very sample-efficient |
| NLL mixing (alpha) | **0** (pure KD loss) | GKD — don't mix with SFT after DPO |
| Dropout | **disabled** | GKD standard |
| Gradient checkpointing | **enabled** | 262K vocab |

## Current vs Recommended

| Aspect | Current Pipeline | Recommended for KD |
|--------|-----------------|-------------------|
| Training signal | SFT (hard labels) → DPO (preference pairs) | On-policy logit KD (soft distributions) |
| Teacher interaction | Offline text generation | Online forward pass for logits |
| Loss function | CE (SFT) / sigmoid (DPO) | Forward KL or JSD |
| Data source | Fixed dataset | Student-generated (on-policy) |
| Teacher model | DeepSeek V4 Flash (diff vocab) | Gemma 4 31B (same 262K vocab) |

## Post-DPO Considerations

- **Must use KL loss, not CE** — cross-entropy after DPO breaks alignment (NVIDIA QAD)
- KD will shift student distribution toward teacher — may partially undo DPO
- **Mitigation**: low LR (1e-5 to 1e-4), few steps (200-500), monitor reward accuracy
- Optional: lightweight DPO refresher (1 epoch) after KD

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 115x capacity gap degrades quality | **High** | Use TAID, or multi-stage 31B→3B→270M |
| KD overwrites DPO alignment | Medium | KL loss, low LR, DPO refresher after |
| OOM with 262K vocab logits | Medium | Batch=1-2, top-K logits, Liger kernel |
| NF4 teacher degrades soft labels | Low | INT8 if concerned (~31GB, still fits) |
| Student can't absorb teacher knowledge | Medium-High | Task-specific KD, not general capability |

## Key Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| GKD (Agarwal et al.) | ICLR 2024 | General framework: on-policy + flexible divergence |
| TAID (Sakana AI) | ICLR 2025 | Handles large capacity gaps via adaptive interpolation |
| DistiLLM (Ko et al.) | ICML 2024 | Skew KL, best for sub-1B students |
| MiniLLM (Gu et al.) | ICLR 2024 | Reverse KL via policy gradient |
| Gemma 2/3 Tech Reports | 2024-25 | Google's production KD recipe |
| LFM2 (Liquid AI) | 2025 | Decoupled Top-K (K=32) for sub-1B |
| NVIDIA QAD | 2025 | Post-alignment KD must use KL, not CE |

## Self-Learning Modes: Hint-Based vs Pure On-Policy Generation

In on-policy distillation / self-learning, there are fundamentally different ways the student generates training data. The key axis is **whether the model gets hints during generation**.

### Mode 1: Pure On-Policy (no hints)

Student generates freely, signal comes after generation via filtering or scoring.

| Method | Signal Source | How It Works |
|--------|-------------|--------------|
| **ReST-EM** (Google DeepMind, TMLR 2024) | Binary filter | Temperature sample 32-64x per problem, keep correct, retrain from base model. Iterate 2-3x. |
| **RFT** (Yuan et al., 2023) | Binary filter | Single iteration of ReST-EM. |
| **SPIN** (Chen et al., ICML 2024) | Distribution matching | GAN-style self-play: model tries to match human data distribution via DPO. No reasoning. |
| **V-STaR** (Hosseini et al., COLM 2024) | Verifier ranking | STaR + DPO-trained verifier on both correct AND incorrect solutions. Best-of-N at inference. |

**Limitation:** model only learns from problems it can already solve. Bootstrapping dead end.

### Mode 2: Answer-Hinted Generation (Rationalization)

For problems the model fails, **provide the correct answer as a hint** and ask it to generate reasoning backward from the answer. Then **strip the hint** and train as if the model generated it unaided.

**The key method: STaR** (Zelikman et al., NeurIPS 2022)

```python
for iteration in range(N):
    for question, correct_answer in dataset:
        # Step 1: Pure generation attempt
        rationale = model.generate(question)
        if rationale.answer == correct_answer:
            keep(question, rationale)  # correct: keep as-is

        # Step 2: RATIONALIZATION (the hint step)
        else:
            hinted_rationale = model.generate(
                question + f"The answer is {correct_answer}. Explain why."
            )
            if hinted_rationale.answer == correct_answer:
                keep(question, hinted_rationale)  # hint stripped for training

    model = finetune(base_model, kept_rationales)
```

**Why it works:** Generating `p(rationale | question, answer)` is much easier than `p(rationale | question)` — the model knows the destination and just finds a path. Without rationalization, STaR covers ~69.7% of CQA data; with it, ~86.7%.

**Risk:** ReST-EM authors explicitly rejected rationalization — found it produces "substantial increase in false positive solutions that result in correct answer but with incorrect reasoning." Temperature sampling (brute-force 32-64 samples) is cleaner but more compute-heavy.

**Results:** GPT-J 6B with STaR matches GPT-3 175B fine-tuned on CommonsenseQA (72.5% vs 73.0%) — 30x parameter efficiency.

### Mode 3: Teacher-Mixed Generation (token-level hints)

Teacher actively intervenes during student generation, not just scoring after.

| Method | Hint Mechanism | When to Use |
|--------|---------------|-------------|
| **MiniLLM** (Gu et al., ICLR 2024) | 20% chance of sampling teacher's token at each step | Reverse KL mode-seeking distillation |
| **TGPO** (Liu et al., 2025) | Teacher predictions as SFT targets at student-generated prefixes | Large student-teacher gap (e.g., 4B→270M) |
| **DaD-DAgger** (Pozzi et al., 2025) | Teacher soft labels + hard labels at student-visited states | Classic imitation learning applied to LLMs |

**TGPO is most relevant for our setup** — specifically designed for large policy divergence where standard GKD/reverse KL fails (uninformative negative feedback). Teacher provides dense directional guidance at each position in the student's trajectory.

### Mode 4: Self-Distillation (no external teacher)

| Method | Mechanism |
|--------|-----------|
| **OPSD** (Zhao et al., 2025) | Same model as teacher+student. Teacher sees ground-truth answer (privileged info), student doesn't. Train student to match teacher's distribution. |
| **Self-Rewarding** (Meta, ICML 2024) | Model judges its own outputs via LLM-as-a-Judge. DPO on self-scored pairs. Requires large models (70B+). |
| **Quiet-STaR** (Zelikman et al., 2024) | Model learns to generate internal thoughts at every token. REINFORCE on helpful thoughts. General reasoning, not task-specific. |

### Mode 5: GKD (token-level distributional hints, post-generation)

This is what `train_gkd_gemma3.py` implements. Student generates on-policy, teacher provides full softmax distribution at each position **after** generation. Dense signal but no intervention during generation.

### Decision Matrix for Our Pipeline

| Method | Requires White-Box Teacher? | Handles Vocab Mismatch? | Signal from Failed Problems? | Best For |
|--------|---------------------------|------------------------|------------------------------|----------|
| ReST-EM | No (verifier only) | N/A | No | Clean data, brute-force coverage |
| STaR rationalization | No (ground truth only) | N/A | **Yes** (key advantage) | Bootstrapping from hard problems |
| GKD | Yes (logits) | Same vocab only | Partial (on-policy exposure) | Dense distributional signal |
| TGPO | Yes (logits) | Same vocab only | **Yes** (directional guidance) | Large capacity gaps |
| DPO (current pipeline) | No (LLM judge) | **Yes** (vocab-agnostic) | Via ranking | Cross-vocab distillation |

### Practical Implications for Gemma3 270M

1. **STaR rationalization** could complement current pipeline: for QA pairs the 270M model gets wrong, re-generate reasoning with the answer as hint, strip hint, add to training data
2. **TGPO** is worth investigating for the GKD training loop — addresses the 4B→270M capacity gap where standard forward KL degrades
3. **V-STaR verifier** could improve inference: train a DPO verifier to rank multiple candidate answers
4. **Self-Rewarding / Quiet-STaR** — 270M too small for self-judgment, skip these

## Implementation Roadmap

1. **Verify Gemma 4 31B teacher quality** — eval on 600 test pairs (in progress, window 3)
2. **Choose approach**: TAID (handles gap) vs GKD forward KL (simpler) vs multi-stage
3. **Implement training loop** — ~300 lines, custom or vendored from trl v0.12.2
4. **Load teacher NF4** on 1 GPU, student on same GPU
5. **Train 200-500 steps** with on-policy generation
6. **Eval** — RAGAS on test set, compare to SFT baseline (0.509) and DPO best (0.532)
7. **Optional DPO refresher** if alignment regressed

# PRD: Generate DPO Training Data on Node 01

## Goal
Generate preference pairs (chosen/rejected) from the SFT-trained Gemma3 270M checkpoint for DPO training. Uses on-policy rollouts scored by an LLM-as-judge.

## Background
- SFT dataset: `qa_pairs_chunked_train.jsonl` (~75K question/context/answer triplets)
- SFT checkpoint: `model_weights/gemma3-270m/hf_ckpts/checkpoint-500/`
- Target: 40-50K usable DPO pairs (after filtering)
- Research references: Nemotron 3 Nano found 10K pairs sufficient; LFM2 used 700K for 350M model

---

## Pipeline overview

```
qa_pairs_chunked_train.jsonl
        |
        v
[1. generate_dpo_rollouts.py]  — vLLM server on node 01, 5 rollouts per prompt, temp=1.0
        |
        v
  dpo_rollouts.jsonl  (75K lines, each with 5 rollouts)
        |
        v
[2. judge_dpo_rollouts.py]  — DeepSeek V4 Flash on node 01, picks best/worst per question
        |
        v
  dpo_rollouts_judged.jsonl  (75K lines, with judgment attached)
        |
        v
[3. build_dpo_pairs.py]  — extract chosen/rejected, filter all_bad/all_good, format for trl
        |
        v
  dpo_pairs_train.jsonl  (~40-50K usable pairs)
```

---

## Step 1: Generate rollouts (`generate_dpo_rollouts.py`)

### Infrastructure
- Serve Gemma3 270M SFT checkpoint via vLLM on rh-h100-01
- Use `--data-parallel-size 8` (model is ~536MB, fits trivially on each GPU)
- `--tensor-parallel-size 1` (no need to shard a 270M model)

### vLLM serve command
```bash
# On rh-h100-01 in tmux
vllm serve model_weights/gemma3-270m/hf_ckpts/checkpoint-500 \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --port 8001  # avoid conflict with DeepSeek on 8000
```

### Script behavior
- Read `qa_pairs_chunked_train.jsonl`
- For each row, build the chat prompt (system + user with context + question)
- Send to vLLM with `n=5, temperature=1.0, max_tokens=1024`
- vLLM returns 5 completions in a single request (native `n` parameter)
- Write one JSONL line per question with all 5 rollouts attached
- Resume support: skip questions already in output file

### Output schema (per line)
```json
{
  "article_id": 871,
  "title": "Amstrad CPC",
  "question": "How did the CPC's video hardware work...",
  "golden_answer": "The original CPC used a Motorola 6845...",
  "context": "The original CPC video hardware supports...",
  "rollouts": ["response_0", "response_1", "response_2", "response_3", "response_4"]
}
```

### Generation parameters
| Parameter | Value | Rationale |
|---|---|---|
| temperature | 1.0 | maximize diversity across rollouts |
| n | 5 | 5 rollouts per prompt (LFM2 used 5) |
| max_tokens | 1024 | match SFT training max seq len |
| top_k | -1 | disabled (pure temperature sampling) |
| top_p | 1.0 | disabled (pure temperature sampling) |

### Concurrency
- Use async OpenAI client against `http://localhost:8001/v1`
- Semaphore: 50-100 concurrent requests (vLLM handles batching internally)
- Model name for API: path to checkpoint (vLLM auto-names from model path)

### Expected throughput
- 270M model on 8x H100 with data parallel = extremely fast
- 75K questions x 5 rollouts = 375K generations
- Estimate: 10-30 minutes total

---

## Step 2: Judge rollouts (`judge_dpo_rollouts.py`)

Already implemented. Uses DeepSeek V4 Flash (self-hosted on node 01 port 8000).

### Judge approach
- Sends all 5 rollouts per question in a single prompt
- Judge sees full response including `<reasoning>...</reasoning>` tags
- Picks `best_idx` and `worst_idx`
- Sets `all_bad=true` if no rollout is correct (skip for DPO)
- Sets `all_good=true` if all rollouts are equivalent (skip for DPO)

### Judge criteria (from system prompt)
- Does the reasoning hallucinate facts not in the context?
- Does the reasoning logically lead to the final answer?
- Does the reasoning make sense given the context?
- Is the final answer factually correct compared to the reference?
- Is the response complete or does it cut off mid-sentence?

### Configuration for self-hosted
```python
BASE_URL = "http://localhost:8000/v1"  # DeepSeek on node 01
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
MAX_CONCURRENT = 10  # self-hosted limit
```

### Output schema (per line)
Same as input + `judgment` field:
```json
{
  "...same fields...",
  "judgment": {
    "best_idx": 0,
    "worst_idx": 4,
    "all_bad": false,
    "all_good": false,
    "explanation": "Response 0 correctly identifies..."
  }
}
```

### Expected throughput
- 75K questions at MAX_CONCURRENT=10
- Estimate: 30-60 minutes

---

## Step 3: Build DPO pairs (`build_dpo_pairs.py`)

### Filtering
- Skip rows where `all_bad=true` (no correct rollout — no learning signal)
- Skip rows where `all_good=true` (no contrast — no learning signal)
- Skip rows where `best_idx == worst_idx` (judge error)
- Skip rows where `best_idx` or `worst_idx` is null

### Output schema (per line, trl DPOTrainer format)
```json
{
  "prompt": "<chat-formatted prompt with system + user>",
  "chosen": "<full text of best rollout including reasoning>",
  "rejected": "<full text of worst rollout including reasoning>"
}
```

### Prompt format
Use the same chat template as SFT training:
```
system: "Answer the question using the provided context."
user: "Context:\n{context}\n\nQuestion: {question}"
```

### Expected yield
- Prototype showed 4/5 (80%) questions produce usable pairs
- 75K x 0.8 = ~60K pairs (conservative: ~40-50K after edge cases)
- Well within the 10K-50K range that Nemotron found sufficient

---

## Execution plan (on rh-h100-01)

### Prerequisites
- DeepSeek V4 Flash already serving on port 8000
- Reserve all 8 GPUs (DeepSeek uses all 8, Gemma3 vLLM needs separate GPUs OR run sequentially)

### Option A: Sequential (simpler, recommended)
1. Tear down DeepSeek temporarily
2. Serve Gemma3 270M on all 8 GPUs (DP8) on port 8001
3. Run `generate_dpo_rollouts.py` → `dpo_rollouts.jsonl`
4. Tear down Gemma3 server
5. Restart DeepSeek V4 Flash on port 8000
6. Run `judge_dpo_rollouts.py dpo_rollouts.jsonl` → `dpo_rollouts_judged.jsonl`
7. Run `build_dpo_pairs.py dpo_rollouts_judged.jsonl` → `dpo_pairs_train.jsonl`

### Option B: Use DeepSeek API for judging (no server juggling)
1. Serve Gemma3 270M on all 8 GPUs
2. Run `generate_dpo_rollouts.py` → `dpo_rollouts.jsonl`
3. Tear down Gemma3 server
4. Run `judge_dpo_rollouts.py` against public DeepSeek API (MAX_CONCURRENT=50)
5. Run `build_dpo_pairs.py`

### File locations (on node 01)
```
~/rawhad/small_lm/qa/
├── qa_pairs_chunked_train.jsonl     # input (scp from local if needed)
├── dpo_rollouts.jsonl               # step 1 output
├── dpo_rollouts_judged.jsonl        # step 2 output
├── dpo_pairs_train.jsonl            # step 3 output (final)
```

---

## Scripts to implement

| Script | Status | Description |
|---|---|---|
| `generate_dpo_rollouts.py` | TODO | vLLM client, async, n=5 per prompt, resume support |
| `judge_dpo_rollouts.py` | DONE | LLM-as-judge, picks best/worst idx |
| `build_dpo_pairs.py` | TODO | Filter + format into trl DPO format |

---

## Open questions
- Should we use the DeepSeek public API for judging (faster, costs money) or self-hosted (free, slower)?
- Exact vLLM flags for Gemma3 270M — may need `--dtype float16` or `--dtype bfloat16`
- Should `build_dpo_pairs.py` also split into train/test?

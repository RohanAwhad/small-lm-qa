# qa — Wikipedia QA Dataset Pipeline

## Structure
- `download_wikipedia.py` — downloads ALL ~6.4M Wikipedia English articles from HF datasets server API to JSONL (resumable, chunked concurrency)
- `utils/wikipedia_loader.py` — shared module: loads articles from local `wikipedia_en.jsonl` by article_id or sequentially
- `generate_qa.py` — generates 15 QA pairs (5 each easy/medium/hard) per article using DeepSeek V4 Flash; loads articles from local JSONL
- `generate_chunked_qa.py` — chunked retrieval-augmented QA: recursive-splits articles into <=512 token chunks, BM25-retrieves top-3 per pair, regenerates answers with reasoning. Resume support. Output feeds `train_hf_gemma3.py`.
- `generate_deepresearch_qa.py` — three-pass deep research QA: article→rubric tree→question→grounded reference answer (QUEST-inspired)
- `generate_deepresearch_qa_multi.py` — multi-article agentic pipeline: chunking→hybrid search index→entity extraction→LLM exploration→synthesis
- `split_train_test.py` — splits `qa_pairs.jsonl` into `qa_train.json` / `qa_test.json` (test capped at 200 per difficulty, seed=42)
- `gemma3.py` — Gemma3 270M model definition in tinygrad (config, RoPE, attention, MLP, weight loading)
- `train_gemma3.py` — fine-tune Gemma3 270M on QA pairs via tinygrad (gradient accumulation, checkpointing)
- `train_hf_gemma3.py` — fine-tune Gemma3 270M via HF Transformers (`unsloth/gemma-3-270m-it`)
- `validate_multi_qa.py` — RAGAS-style validation for multi-article QA: faithfulness (answer claims vs source articles) + context relevance (exploration log focus). Deterministic sentence splitting, resume support.
- `src/evals/` — **primary eval** (modular): decompose → classify → score pipeline. See `src/evals/AGENTS.md` for details.
  - `src/evals/decompose.py` — answer text → atomic claims via LLM (tenacity retry)
  - `src/evals/classify.py` — agent claims vs ref claims → SUPPORTED/CONTRADICTED/UNSUPPORTED verdicts (tenacity retry)
  - `src/evals/scoring.py` — verdicts → P/R/F1 (pure function)
  - `src/evals/run.py` — orchestration: `evaluate_single` + `evaluate_all` with CLI
- `evaluate_ragas.py` — **legacy eval**: monolithic RAGAS eval, superseded by `src/evals/`
- `verify_golden.py` — 4-vote LLM judge against Wikipedia article; unanimous = correct. No resume support (clears output on rerun).
- `src/generation/` — **primary generation** (modular): load → messages → engine → output. See `src/generation/AGENTS.md` for details.
  - `src/generation/constants.py` — SamplingParams frozen dataclass, system prompt, message template
  - `src/generation/utils.py` — `generate_messages(context, question)` message builder
  - `src/generation/hf_engine.py` — batched HF Transformers inference
  - `src/generation/vllm_engine.py` — async vLLM API inference with tenacity retry
  - `src/generation/run.py` — orchestration: QAPair validation, engine dispatch, CLI
- `generate_gemma_answers.py` — **legacy**: Ollama-based answer generation
- `generate_hf_answers.py` — **legacy**: standalone HF inference, superseded by `src/generation/`
- `generate_vllm_answers.py` — **legacy**: standalone vLLM inference, superseded by `src/generation/`
- `summarize_scores.py` — pretty-prints RAGAS eval scores (P/R/F1, claim counts) grouped by difficulty
- `generate_dpo_rollouts.py` — generates N rollouts per prompt via vLLM server (async, resume support). Dynamic `max_tokens` per prompt: `min(2048 - prompt_tokens - 100, 1024)`, skips prompts with <64 output tokens. Output feeds `judge_dpo_rollouts.py`.
- `judge_dpo_rollouts.py` — LLM-as-judge for DPO: sends all rollouts per question to DeepSeek Flash, picks best/worst idx. Evaluates reasoning + answer quality. Resume support. Config via env vars (`BASE_URL`, `DEEPSEEK_MODEL`, `MAX_CONCURRENT`).
- `build_dpo_pairs.py` — filters judged rollouts (skip all_bad/all_good/invalid), formats into trl DPOTrainer JSONL (prompt/chosen/rejected message lists).
- `train_dpo_gemma3.py` — DPO training via trl DPOTrainer on top of SFT checkpoint. Multi-GPU via `torchrun --nproc_per_node=8`. Config: beta=5.0, LR=1e-6, effective batch=64.
- `train_gkd_gemma3.py` — on-policy knowledge distillation (GKD): Gemma3 4B teacher → 270M student. Forward KL on student-generated sequences. Handles vocab size mismatch (student 262144 vs teacher 262208) by slicing.
- `evaluate_qa.py` — **legacy**: 1-5 judge format, superseded by `evaluate_ragas.py`
- `compare_gemma.py` — **legacy**: evaluates Gemma3 against DeepSeek golden using old 1-5 format
- `tests/test_schema.py` — validates JSONL output files against expected schema (read-only, no API)
- `tests/test_generate_evaluate.py` — QA gen + eval via local Gemma (req Ollama)
- `tests/test_e2e.py` — end-to-end Gemma answer pipeline (req Ollama)

## Commands
```bash
# Download all Wikipedia English articles (~6.4M, resumable)
uv run python download_wikipedia.py [N]  # N optional, defaults to ALL

# Generate QA from N articles (resumes — skips already-processed article_ids)
uv run python generate_qa.py N

# Generate Gemma3 answers (streams output line-by-line, tenacity retry on failures)
uv run python generate_gemma_answers.py [input.jsonl] [-o output.jsonl]

# RAGAS eval: claim-based P/R/F1 (modular, no resume — errors if output exists)
uv run python -m src.evals.run <input.jsonl> -o <output.jsonl> [--overwrite]

# Legacy RAGAS eval (has resume — skips already-evaluated pairs)
uv run python evaluate_ragas.py [input.jsonl] --reference qa_pairs.jsonl [-o output.jsonl]

# Validate multi-article QA: faithfulness + context relevance (has resume)
uv run python validate_multi_qa.py [input.jsonl] [-o output.jsonl]

# 4-vote golden verification against Wikipedia (no resume — overwrites output)
uv run python verify_golden.py [input.jsonl] [-o output.jsonl]

# Summarize RAGAS eval scores
uv run python summarize_scores.py [path/to/ragas_eval.jsonl]

# DPO data generation (run on rh-h100-01)
# Step 1: Generate rollouts via vLLM-served model
.venv/bin/python generate_dpo_rollouts.py qa_pairs_chunked_train.jsonl -o dpo_rollouts.jsonl --base-url http://localhost:8001/v1

# Step 2: Judge rollouts via DeepSeek Flash (self-hosted)
BASE_URL=http://localhost:8000/v1 DEEPSEEK_MODEL=deepseek-ai/DeepSeek-V4-Flash DEEPSEEK_API_KEY=dummy MAX_CONCURRENT=20 \
  .venv/bin/python judge_dpo_rollouts.py dpo_rollouts.jsonl -o dpo_rollouts_judged.jsonl

# Step 3: Build DPO pairs
.venv/bin/python build_dpo_pairs.py dpo_rollouts_judged.jsonl -o dpo_pairs_train.jsonl

# Tests (standalone scripts, not pytest)
uv run python tests/test_schema.py          # validates existing JSONL files
uv run python tests/test_generate_evaluate.py  # req Ollama + gemma3:270m
uv run python tests/test_e2e.py                # req Ollama + gemma3:270m
```

## Pipeline (current)
1. `generate_qa.py N` → `qa_pairs.jsonl` (golden QA pairs)
2. `generate_gemma_answers.py qa_test.json` → `qa_test_gemma.jsonl` (agent answers)
3. `evaluate_ragas.py qa_test_gemma.jsonl --reference qa_pairs.jsonl` → `qa_test_gemma_ragas_eval.jsonl`
4. `summarize_scores.py qa_test_gemma_ragas_eval.jsonl`
5. (optional) `verify_golden.py qa_pairs.jsonl` → `qa_pairs_verified.jsonl`

## RAGAS eval: cached artifacts
- `qa_reference_claims.jsonl` — golden answer claims, keyed by `article_id::question`
- `qa_article_claims.jsonl` — Wikipedia article claims, keyed by `article_id`
- These are regenerated as needed; delete to force re-decomposition

## Dependencies
- `uv` for package management (Python 3.12)
- `DEEPSEEK_API_KEY` env var required for DeepSeek API calls
- `LOGGING_LEVEL` env var controls log verbosity (INFO stderr, DEBUG to `logs/`)
- Ollama tests gracefully skip if `gemma3:270m` not pulled or unreachable

## DeepSeek API quirks
- Model: `deepseek-v4-flash` (thinking enabled by default)
- Does NOT support `json_schema` — only `response_format={"type": "json_object"}`
- Prompt must contain the word `json` when using `json_object`
- Thinking consumes output token budget — don't set `max_tokens` too low
- Thinking tokens available via `choice.message.reasoning_content`; stored by `generate_chunked_qa.py` (used in training), discarded by other scripts

## Training config (Gemma3 270M)
- Model: `unsloth/gemma-3-270m-it` (268M params, 262K vocab)
- Google's official recommended LR: **5e-5** constant (for both full FT and LoRA)
- Current config: LR=3e-5, constant scheduler, no warmup
- Effective batch size: 64 (bs=16, grad_accum=4)
- Max seq len: 1024 tokens (covers 87.5% of chunked QA data)
- Attention: SDPA (not eager — ~2x faster on H100)
- Gradient checkpointing: enabled (262K vocab logits use ~32GB)
- Dynamic padding: pad to batch max, not max_seq_len (saves compute on short seqs)
- Overfitting indicator: training loss below 0.2 suggests overfitting (per Unsloth)
- torch.compile: broken on transformers 5.9.0 — do not use
- Full fine-tune preferred over LoRA for 270M (model is only ~536MB, LoRA saves nothing)
- Training format: `system: "Answer the question using the provided context." | user: context+question | assistant: <reasoning>...</reasoning> + answer`
- Filters: drop pairs with reasoning > 512 tokens, drop pairs with total seq > 1024 tokens
- Wandb project: `small-lm-qa`

## Remote GPU node (rh-h100-01)
- SSH: `ssh rh-h100-01`
- Project path: `/home/lab/rawhad/small_lm/qa/`
- Checkpoints: `model_weights/gemma3-270m/hf_ckpts/checkpoint-{500,1000,...}/` and `final/`
- **Use `.venv/bin/python`** — do NOT use `uv run` on the node (breaks NCCL/torch linkage)
- Reserve GPUs before use: `ssh rh-h100-01 'gpu reserve --gpu-ids 1 --user "Rohan" --note "reason" --duration 2h'`
- Training runs on GPU 0; use GPU 1+ for eval: `export CUDA_VISIBLE_DEVICES=1`
- Wandb project: `small-lm-qa`

### Training workflow (end-to-end)
```bash
# 1. Reserve GPU
ssh rh-h100-01 'gpu reserve --gpus 1 --user "Rohan" --note "gemma3 training" --duration 6h'

# 2. Sync code and data to remote
cd ~/1_Projects/personal_projects/small_lm/qa
git push
ssh rh-h100-01 'cd ~/rawhad/small_lm/qa && git fetch origin && git reset --hard origin/main'
scp qa_pairs_chunked_train.jsonl rh-h100-01:~/rawhad/small_lm/qa/

# 3. Start training in tmux (use .venv/bin/python, NOT uv run)
WANDB_KEY=$(echo $WANDB_API_KEY)
ssh rh-h100-01 "tmux new-session -d -s train \"cd ~/rawhad/small_lm/qa && export CUDA_VISIBLE_DEVICES=0 && export WANDB_API_KEY=$WANDB_KEY && export WANDB_PROJECT=small-lm-qa && .venv/bin/python train_hf_gemma3.py 2>&1 | tee logs/train_chunked_qa.log\""

# 4. Monitor
ssh rh-h100-01 'tail -5 ~/rawhad/small_lm/qa/logs/train_chunked_qa.log'

# 5. Pull checkpoint to local
scp rh-h100-01:~/rawhad/small_lm/qa/model_weights/gemma3-270m/hf_ckpts/checkpoint-500/{model.safetensors,config.json,tokenizer.json,tokenizer_config.json,generation_config.json,chat_template.jinja} model_weights/gemma3-270m/hf_ckpts/checkpoint-500/

# 6. Convert to Ollama (requires llama.cpp cloned at /tmp/llama.cpp)
uv run python /tmp/llama.cpp/convert_hf_to_gguf.py model_weights/gemma3-270m/hf_ckpts/checkpoint-500/ --outtype q8_0 --outfile model_weights/gemma3-270m/gemma3-270m-qa-step500.gguf
# Note: must patch /tmp/llama.cpp/conversion/base.py — comment out `assert max(tokenizer.vocab.values()) < vocab_size` (2 lines) for Gemma3's 262K vocab
ollama create gemma3-270m-qa:step500 -f model_weights/gemma3-270m/Modelfile

# 7. Release GPU
ssh rh-h100-01 'gpu release'
```

### Remote eval workflow (generate_hf_answers.py)
```bash
# On node — batch HF inference, much faster than local Ollama
ssh rh-h100-01
cd ~/rawhad/small_lm/qa
export CUDA_VISIBLE_DEVICES=1

# Baseline
.venv/bin/python generate_hf_answers.py qa_test.json -o qa_test_hf_baseline.jsonl -m unsloth/gemma-3-270m-it

# Fine-tuned checkpoint
.venv/bin/python generate_hf_answers.py qa_test.json -o qa_test_hf_step500.jsonl -m model_weights/gemma3-270m/hf_ckpts/checkpoint-500

# Pull results to local
scp rh-h100-01:~/rawhad/small_lm/qa/qa_test_hf_*.jsonl .

# RAGAS eval locally (uses DeepSeek API)
uv run python evaluate_ragas.py qa_test_hf_baseline.jsonl --reference qa_pairs.jsonl -o qa_test_hf_baseline_ragas_eval.jsonl
uv run python evaluate_ragas.py qa_test_hf_step500.jsonl --reference qa_pairs.jsonl -o qa_test_hf_step500_ragas_eval.jsonl
uv run python summarize_scores.py qa_test_hf_baseline_ragas_eval.jsonl
uv run python summarize_scores.py qa_test_hf_step500_ragas_eval.jsonl
```

## Self-hosted DeepSeek V4 Flash
- **ASK ROHAN before deploying/restarting** — do not start vLLM on any node without explicit permission
- Model: `deepseek-ai/DeepSeek-V4-Flash` on all 8 GPUs via vLLM
- Currently deployed on rh-h100-01: `http://localhost:8000/v1` (from node 01 itself)
- Also deployed on rh-h100-11: `http://localhost:8000/v1` (internal IP: `10.241.128.16`)
- Model name for API: `deepseek-ai/DeepSeek-V4-Flash`
- Recommended: `--max-model-len 16384 --max-num-seqs 256` (judge prompts are <6K tokens; 16K is plenty; can push `--max-num-seqs` to 1024 for higher throughput)
- Can replace DeepSeek API for RAGAS eval — set `BASE_URL` and `DEEPSEEK_MODEL` in eval scripts
- Use `MAX_CONCURRENT = 256` for DPO judging workloads (high concurrency, short outputs)

### RAGAS eval via self-hosted model (on node 01)
```bash
# Run from node 01 — uses DeepSeek V4 Flash on node 01 as LLM judge
ssh rh-h100-01
cd ~/rawhad/small_lm/qa
DEEPSEEK_API_KEY=dummy .venv/bin/python -c "
import evaluate_ragas, asyncio, os
from pathlib import Path
evaluate_ragas.BASE_URL = 'http://10.241.128.23:8000/v1'  # node 01 internal IP
evaluate_ragas.DEEPSEEK_MODEL = 'deepseek-ai/DeepSeek-V4-Flash'
evaluate_ragas.MAX_CONCURRENT = 10
os.environ['DEEPSEEK_API_KEY'] = 'dummy'
asyncio.run(evaluate_ragas.main(
    Path('qa_test_hf_step1031.jsonl'),
    Path('qa_test_hf_step1031_ragas_selfhosted.jsonl'),
    Path('qa_pairs.jsonl'),
))
"
```
- Full deployment details (flags, setup, troubleshooting): see **vLLM Model Deployment** skill

### Deploying on any node (venv setup)
```bash
# 1. Create venv dir and HF cache on NVMe (check `df -h` for the mount path)
ssh rh-h100-XX 'mkdir -p ~/rawhad/venvs && mkdir -p /mnt/nvme0n1/rawhad/hf_cache'

# 2. Create venv and install vllm
ssh rh-h100-XX 'cd ~/rawhad/venvs && uv venv vllm_venv --python 3.12'
ssh rh-h100-XX 'source ~/rawhad/venvs/vllm_venv/bin/activate && uv pip install vllm'

# 3. Deploy in tmux
ssh rh-h100-XX "tmux new-session -d -s vllm 'export HF_HOME=/mnt/nvme0n1/rawhad/hf_cache && source ~/rawhad/venvs/vllm_venv/bin/activate && vllm serve deepseek-ai/DeepSeek-V4-Flash --trust-remote-code --kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel --tensor-parallel-size 8 --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 --max-model-len 16384 --max-num-seqs 256 2>&1 | tee ~/rawhad/vllm_deepseek_v4_flash.log; sleep infinity'"

# 4. Monitor
ssh rh-h100-XX 'tail -f ~/rawhad/vllm_deepseek_v4_flash.log'
```
- NVMe mount path varies by node — check `df -h | grep nvme` first
- Requires full node (8 GPUs) — reserve all before deploying

## Code conventions
- All scripts use `asyncio.run(main())` — no sync entrypoints (training scripts are sync)
- Articles loaded from local `wikipedia_en.jsonl` via `wikipedia_loader.py` (pre-downloaded with `download_wikipedia.py`; no longer fetched from HF API on-the-fly)
- `article_id` maps to row index for re-fetching
- `*.jsonl`, `logs/`, `model_weights/`, `play.py`, and `wandb/` are gitignored — regenerated artifacts
- `play.py.output*` and `play.py.output.json` are NOT gitignored — clean up after use
- `.json` files (`qa_train.json`, `qa_test.json`) ARE tracked in git — these are committed splits
- `docs/` has quantization research notes (not scripts)
- Tests are standalone scripts (NOT pytest) — run with `uv run python tests/test_*.py`
- Test fixtures live in `tests/fixtures/` (e.g. `article_0.json`)
- No linting, formatting, or CI configured — no ruff/black/mypy/pytest to run
- Gemma3 270M answers degrade with full article text; needs truncation for large articles
- Pydantic models use `""` default for optional `reasoning` fields (LLM sometimes omits it)
- Training backends: tinygrad (`train_gemma3.py` + `gemma3.py`), HF Transformers SFT (`train_hf_gemma3.py`), DPO (`train_dpo_gemma3.py`), GKD (`train_gkd_gemma3.py`)
- `README.md` has the pipeline overview; this file has operational details and quirks

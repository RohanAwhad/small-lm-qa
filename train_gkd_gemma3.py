"""On-policy knowledge distillation (GKD) for Gemma3 270M.

Teacher: Gemma3 4B-it (bf16, GPU 1)
Student: Gemma3 270M DPO checkpoint (bf16, GPU 0)
Loss: Forward KL on student-generated sequences (lambda=1.0)

Vocab alignment note:
  Student lm_head outputs 262,144 logits. Teacher lm_head outputs 262,208 logits.
  Both share the same tokenizer (262,145 tokens, verified by text at every ID).
  The teacher's extra 64 logit positions (262,144-262,207) are padding to a GPU-
  friendly multiple of 64; only id=262,144 is defined (<image_soft_token>).
  We slice both to [:262,144] before computing KL — safe because all extra
  positions are at the end and unused in our QA data.

References:
  - GKD: Agarwal et al., ICLR 2024 (arXiv:2306.13649)
  - Gemma 2/3 post-training distillation recipe

Usage (on rh-h100-01):
    .venv/bin/python train_gkd_gemma3.py
"""

import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Config
# ============================================================================

STUDENT_MODEL = "model_weights/gemma3-270m/dpo-4gpu/checkpoint-955"
TEACHER_MODEL = "google/gemma-3-4b-it"
TRAIN_DATA = "qa_pairs_chunked_train.jsonl"
OUTPUT_DIR = "model_weights/gemma3-270m/gkd"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 32  # effective bs = 64
LR = 1e-4
NUM_STEPS = 500
SAVE_EVERY = 100
LOGGING_STEPS = 1
TEMPERATURE = 1.0
STUDENT_GPU = "cuda:0"
TEACHER_GPU = "cuda:1"

SYSTEM_PROMPT = "Answer the question using the provided context."

IS_RANK_0 = int(os.environ.get("LOCAL_RANK", 0)) == 0

# ============================================================================
# Loss
# ============================================================================

def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Token-level forward KL: KL(teacher || student) on non-padding positions."""
    # Align vocab sizes (student may have extra tokens like <image_soft_token>)
    min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_logits = student_logits[..., :min_vocab]
    teacher_logits = teacher_logits[..., :min_vocab]

    mask = labels != -100

    s_logits = student_logits[mask].float() / temperature
    t_logits = teacher_logits[mask].float() / temperature

    t_log_probs = F.log_softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)

    # KL(teacher || student) = sum_c p_T(c) * (log p_T(c) - log p_S(c))
    loss = F.kl_div(s_log_probs, t_log_probs, reduction="batchmean", log_target=True)
    return loss * (temperature ** 2)


# ============================================================================
# Data
# ============================================================================

def load_prompts(path: str) -> list[dict]:
    """Load QA pairs — we only need the prompts (context + question), not answers."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_prompt(record: dict, tokenizer: AutoTokenizer) -> str:
    """Format a QA record into a chat prompt (system + user, no assistant)."""
    context = "\n\n".join(record["context_chunks"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {record['question']}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ============================================================================
# Main
# ============================================================================

def main():
    if IS_RANK_0:
        wandb.init(
            project="small-lm-gkd",
            name=f"gkd-4b-teacher-{NUM_STEPS}steps",
            config={
                "student": STUDENT_MODEL,
                "teacher": TEACHER_MODEL,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM_STEPS,
                "effective_bs": BATCH_SIZE * GRAD_ACCUM_STEPS,
                "max_seq_len": MAX_SEQ_LEN,
                "temperature": TEMPERATURE,
                "num_steps": NUM_STEPS,
                "loss": "forward_kl",
                "lambda": 1.0,
            },
        )

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for generation
    print(f"Tokenizer vocab: {tokenizer.vocab_size}")

    # --- Load teacher (frozen, bf16, GPU 1) ---
    print(f"Loading teacher: {TEACHER_MODEL} on {TEACHER_GPU}")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(TEACHER_GPU)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")

    # --- Load student (trainable, bf16, GPU 0) ---
    print(f"Loading student: {STUDENT_MODEL} on {STUDENT_GPU}")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(STUDENT_GPU)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"Student loaded: {sum(p.numel() for p in student.parameters()):,} params")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
    )

    # --- Load and shuffle prompts ---
    all_records = load_prompts(TRAIN_DATA)
    random.shuffle(all_records)
    print(f"Loaded {len(all_records)} prompts from {TRAIN_DATA}")

    # --- Training loop ---
    global_step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    record_idx = 0
    while global_step < NUM_STEPS:
        for micro_step in range(GRAD_ACCUM_STEPS):
            # Get batch of prompts
            batch_records = []
            for _ in range(BATCH_SIZE):
                if record_idx >= len(all_records):
                    random.shuffle(all_records)
                    record_idx = 0
                batch_records.append(all_records[record_idx])
                record_idx += 1

            # Format prompts
            prompts = [format_prompt(r, tokenizer) for r in batch_records]
            MIN_NEW_TOKENS = 64
            prompt_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN - MIN_NEW_TOKENS,
            ).to(STUDENT_GPU)
            prompt_len = prompt_inputs["input_ids"].shape[1]
            max_new = MAX_SEQ_LEN - prompt_len

            # --- Phase 1: Student generates (no grad) ---
            student.eval()
            with torch.no_grad():
                gen_outputs = student.generate(
                    **prompt_inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=0,
                    use_cache=True,
                )
            # gen_outputs: [B, prompt_len + gen_len]
            full_ids = gen_outputs
            full_mask = (full_ids != tokenizer.pad_token_id).long()

            # --- Phase 2: Teacher scores (no grad, on teacher GPU) ---
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=full_ids.to(TEACHER_GPU),
                    attention_mask=full_mask.to(TEACHER_GPU),
                )
                # Shift for next-token prediction, completion only
                teacher_logits = teacher_out.logits[:, prompt_len - 1 : -1, :]

            # --- Phase 3: Student re-scores (WITH grad) ---
            student.train()
            student_out = student(
                input_ids=full_ids,
                attention_mask=full_mask,
            )
            student_logits = student_out.logits[:, prompt_len - 1 : -1, :]

            # --- Phase 4: Compute loss ---
            # Labels: completion tokens, -100 for padding
            completion_ids = full_ids[:, prompt_len:]
            labels = completion_ids.clone()
            labels[completion_ids == tokenizer.pad_token_id] = -100

            loss = forward_kl_loss(
                student_logits,
                teacher_logits.to(STUDENT_GPU),
                labels,
                temperature=TEMPERATURE,
            )
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()

        # --- Optimizer step ---
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        # --- Logging ---
        if global_step % LOGGING_STEPS == 0:
            log_data = {
                "loss": accum_loss,
                "step": global_step,
                "lr": LR,
            }
            print(f"Step {global_step}/{NUM_STEPS} | loss={accum_loss:.4f}")
            if IS_RANK_0:
                wandb.log(log_data, step=global_step)
            accum_loss = 0.0

        # --- Save checkpoint ---
        if global_step % SAVE_EVERY == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            student.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint: {ckpt_dir}")

    # --- Final save ---
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final model: {final_dir}")

    if IS_RANK_0:
        wandb.finish()


if __name__ == "__main__":
    main()

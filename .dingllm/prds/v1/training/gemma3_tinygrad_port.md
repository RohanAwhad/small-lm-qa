# PRD: Port Gemma 3 from HuggingFace Transformers to tinygrad

## Goal
Rewrite Gemma 3 270M (text-only, no vision) in tinygrad for training from scratch, achieving numerical equivalence with HF transformers implementation.

## Target model
- **Model**: Gemma 3 270M architecture (training from scratch, not loading pretrained)
- **Source**: `transformers/models/gemma3/modeling_gemma3.py`
- **Destination**: single-file `gemma3.py` in tinygrad style

## Scope
- **In scope**: text-only Gemma 3 architecture, training from scratch
- **Out of scope (v1)**: vision encoder (SiglipVision), multimodal projector, quantization, inference/generation, KV cache

---

## Architecture: Gemma 3 Text (from HF source)

### Config (270M — from `google/gemma-3-270m` via lmstudio-community mirror)

| Parameter | Value | Notes |
|---|---|---|
| vocab_size | 262,144 | |
| hidden_size | 640 | |
| intermediate_size | 2048 | ~3.2x hidden |
| num_hidden_layers | 18 | |
| num_attention_heads | 4 | |
| num_key_value_heads | 1 | GQA ratio 4:1 (MQA) |
| head_dim | 256 | != hidden_size/num_heads (160) |
| hidden_activation | gelu_pytorch_tanh | NOT SiLU |
| rms_norm_eps | 1e-6 | |
| sliding_window | 512 | Local attention window |
| sliding_window_pattern | 6 | Every 6th layer = full attn |
| max_position_embeddings | 32,768 | |
| query_pre_attn_scalar | 256 | Attention scaling factor |
| rope_theta (global) | 1,000,000 | For full attention layers |
| rope_theta (local) | 10,000 | For sliding attention layers |
| tie_word_embeddings | True | lm_head = embed_tokens.T |
| final_logit_softcapping | None | |
| attn_logit_softcapping | None | |
| attention_bias | False | No bias in QKV/O projections |

Layer type schedule (18 layers):
```
[sliding, sliding, sliding, sliding, sliding, FULL,
 sliding, sliding, sliding, sliding, sliding, FULL,
 sliding, sliding, sliding, sliding, sliding, FULL]
```

### Key architectural differences from tinygrad LLaMA

| Feature | tinygrad LLaMA | Gemma 3 | Delta |
|---|---|---|---|
| Activation | SiLU | gelu_pytorch_tanh | Change activation fn |
| head_dim | hidden_size / n_heads | Explicit (256) | Decouple from hidden_size |
| RMSNorm | `x * weight` | `x * (1.0 + weight)` | +1 offset, weight init=0 |
| Norms per layer | 2 (attn_norm, ffn_norm) | 4 (input, post_attn, pre_ffn, post_ffn) | Add 2 more norms |
| QK norms | Optional (qk_norm param) | Always on (RMSNorm) | Wire up always |
| Embedding | Plain lookup | Scaled by sqrt(hidden_size) | Multiply after embed |
| RoPE | Single theta | Dual theta (local=10k, global=1M) | Per-layer-type freqs |
| Attention mask | Causal only | Causal + sliding window | Alternating mask pattern |
| Attention scaling | 1/sqrt(head_dim) | 1/sqrt(query_pre_attn_scalar) | Use config scalar |
| Tied weights | Optional | lm_head = embed_tokens | Share weight tensor |
| Logit softcapping | None | Optional tanh capping | Implement if config says so |

---

## Implementation Plan

### Phase 1: Primitives

**P1.1 — Gemma3RMSNorm**
```
# Difference: (1.0 + weight) instead of just weight
# Weight initialized to zeros (so starts as identity)
def _norm(x): x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
def forward(x): _norm(x.float()) * (1.0 + weight.float())  # cast back after
```
- Test: random input, compare output vs HF Gemma3RMSNorm

**P1.2 — Gemma3RotaryEmbedding (dual theta)**
```
# Two sets of freqs_cis:
#   - local (sliding_attention): theta=10,000
#   - global (full_attention):   theta=1,000,000
# Each layer type indexes into its own precomputed freqs
```
- Test: compare cos/sin tables at various positions vs HF

**P1.3 — Embedding scaling**
```
# embed_tokens(input_ids) * sqrt(hidden_size)
# Note: HF downcasts to bf16 causing sqrt(3072)=55.4256→55.5
```

**P1.4 — gelu_pytorch_tanh activation**
```
# GELU with tanh approximation
# 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# tinygrad Tensor.gelu() exists — verify it matches pytorch_tanh variant
```
- Test: compare against torch.nn.functional.gelu(x, approximate='tanh')

### Phase 2: Attention

**P2.1 — Gemma3Attention**
```
class Gemma3Attention:
  q_proj: Linear(hidden_size, num_heads * head_dim)      # 640 → 1024
  k_proj: Linear(hidden_size, num_kv_heads * head_dim)   # 640 → 256
  v_proj: Linear(hidden_size, num_kv_heads * head_dim)   # 640 → 256
  o_proj: Linear(num_heads * head_dim, hidden_size)      # 1024 → 640

  q_norm: Gemma3RMSNorm(head_dim)   # per-head norm
  k_norm: Gemma3RMSNorm(head_dim)   # per-head norm

  scaling = query_pre_attn_scalar ** -0.5   # 1/16, not 1/sqrt(head_dim)

  Forward:
    1. Project: q, k, v = q_proj(x), k_proj(x), v_proj(x)
    2. Reshape: [B, S, n_heads, head_dim]
    3. QK norm: q = q_norm(q), k = k_norm(k)
    4. RoPE: apply_rotary_emb(q, k, freqs_cis)  # use layer-specific freqs
    5. GQA: repeat_kv for k,v
    6. Attention: softmax(q @ k.T * scaling) @ v
    7. Output: o_proj(attn_output)
```
- Test: single layer, identical weights, compare output vs HF

**P2.2 — Sliding window mask**
```
# Layer type determined by: layer_types[i]
#   - "sliding_attention" if (i+1) % 6 != 0
#   - "full_attention"    if (i+1) % 6 == 0
#
# Sliding mask: token can only attend to positions within [pos - window, pos]
# Full mask: standard causal mask
```
- Test: generate mask matrices, compare against HF create_sliding_window_causal_mask

### Phase 3: Decoder Layer

**P3.1 — Gemma3DecoderLayer**
```
class Gemma3DecoderLayer:
  self_attn: Gemma3Attention
  mlp: Gemma3MLP
  input_layernorm: Gemma3RMSNorm          # before attention
  post_attention_layernorm: Gemma3RMSNorm  # after attention, before residual add
  pre_feedforward_layernorm: Gemma3RMSNorm # before MLP
  post_feedforward_layernorm: Gemma3RMSNorm # after MLP, before residual add

  Forward:
    residual = x
    x = input_layernorm(x)
    x = self_attn(x, freqs_cis, mask)
    x = post_attention_layernorm(x)
    x = residual + x

    residual = x
    x = pre_feedforward_layernorm(x)
    x = mlp(x)
    x = post_feedforward_layernorm(x)
    x = residual + x
```
- Note: LLaMA has 2 norms (pre-attn, pre-ffn). Gemma 3 has 4 (pre+post for both).

**P3.2 — Gemma3MLP**
```
class Gemma3MLP:
  gate_proj: Linear(hidden_size, intermediate_size)
  up_proj:   Linear(hidden_size, intermediate_size)
  down_proj: Linear(intermediate_size, hidden_size)

  Forward: down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))
```
- Same structure as LLaMA SwiGLU but with gelu_tanh instead of silu

### Phase 4: Full Model

**P4.1 — Gemma3Transformer**
```
class Gemma3Transformer:
  tok_embeddings: Embedding(vocab_size, hidden_size)
  layers: [Gemma3DecoderLayer] * num_hidden_layers
  norm: Gemma3RMSNorm(hidden_size)
  # No separate output/lm_head — tied to tok_embeddings

  # Two sets of freqs_cis (precomputed)
  freqs_cis_local:  precompute(head_dim, max_ctx, theta=10_000)
  freqs_cis_global: precompute(head_dim, max_ctx, theta=1_000_000)

  # Layer type schedule
  layer_types: ["sliding_attention", ..., "full_attention", ...]
  # Pattern: sliding unless (i+1) % 6 == 0

  Forward:
    h = tok_embeddings(tokens) * sqrt(hidden_size)   # scaled embedding
    for i, layer in enumerate(layers):
      freqs = freqs_cis_local if layer_types[i] == "sliding_attention" else freqs_cis_global
      mask = sliding_mask if layer_types[i] == "sliding_attention" else causal_mask
      h = layer(h, freqs, mask)
    h = norm(h)
    logits = h @ tok_embeddings.weight.T   # tied weights
    return logits
```

**P4.2 — Logit softcapping (optional)**
```
# Only if config.final_logit_softcapping is not None:
logits = logits / softcap
logits = tanh(logits)
logits = logits * softcap
```

### Phase 5: Training

**P5.1 — Loss function**
- Cross-entropy loss on shifted logits vs labels
- `logits[..., :-1, :].sparse_categorical_crossentropy(labels[..., 1:])`
- tinygrad has autograd — `.backward()` computes gradients

**P5.2 — Optimizer**
- AdamW (tinygrad has `nn.optim.AdamW`)
- Standard hyperparams: lr, weight_decay, beta1, beta2

**P5.3 — Training loop**
```python
for batch in dataloader:
  logits = model(batch.input_ids)
  loss = cross_entropy(logits[:, :-1], batch.input_ids[:, 1:])
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
```

**P5.4 — Data loading**
- Load tokenized data (pre-tokenized JSONL or use tokenizer on-the-fly)
- Batch, pad/truncate to max_seq_len
- tinygrad doesn't have a DataLoader — write a simple generator

---

## Testing Strategy

### T1: Component-level numerical equivalence
For each component, load identical weights into HF and tinygrad, feed same input, compare output.

| Test | Input | Tolerance (fp32) | Tolerance (bf16) |
|---|---|---|---|
| Gemma3RMSNorm | random [B,S,D] | 1e-6 | 1e-3 |
| gelu_pytorch_tanh | random [B,S,D] | 1e-6 | 1e-3 |
| RoPE cos/sin tables | position_ids | 1e-5 | 1e-2 |
| Attention (single layer) | random + weights | 1e-5 | 1e-2 |
| MLP (single layer) | random + weights | 1e-5 | 1e-2 |
| DecoderLayer | random + weights | 1e-4 | 1e-2 |
| Full model logits | input_ids | 1e-4 | 5e-2 |

### T2: Sliding window correctness
- Verify that local attention layers only attend within window
- Compare attention patterns (if extractable) between HF and tinygrad
- Test with sequence longer than sliding_window to confirm truncation

### T3: Training smoke test
- Forward + backward pass completes without error
- Loss decreases over 10 steps on a small dataset
- Gradient shapes match parameter shapes

---

## File Structure

```
gemma3.py              # Model definition (single file, tinygrad style)
test_gemma3.py         # Numerical equivalence tests
```

## Dependencies
- `tinygrad` (latest)
- `torch` + `transformers` (test-only, for reference model comparison)

## Order of execution
1. **P0** → verify tinygrad gelu matches gelu_pytorch_tanh (play.py experiment)
2. **P1** → primitives + tests for each
3. **P2** → attention + test
4. **P3** → decoder layer + test
5. **P4** → full model + forward pass test
6. **P5** → training loop + loss decreasing smoke test

## Open questions
- Does tinygrad's `Tensor.gelu()` match `gelu_pytorch_tanh`? → **verify first** (P0, play.py)

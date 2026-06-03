"""Gemma 3 270M in tinygrad — training from scratch."""
from dataclasses import dataclass
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.optim import AdamW
import math

# ============================================================================
# Config
# ============================================================================

@dataclass
class Gemma3Config:
    vocab_size: int = 262_144
    hidden_size: int = 640
    intermediate_size: int = 2048
    num_hidden_layers: int = 18
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    sliding_window: int = 512
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 32_768
    query_pre_attn_scalar: float = 256.0
    rope_theta_global: float = 1_000_000.0
    rope_theta_local: float = 10_000.0

    @property
    def layer_types(self) -> list[str]:
        return [
            "full_attention" if (i + 1) % self.sliding_window_pattern == 0 else "sliding_attention"
            for i in range(self.num_hidden_layers)
        ]

# ============================================================================
# P1: Primitives
# ============================================================================

class Gemma3RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = Tensor.zeros(dim)  # init to 0, used as (1 + weight)

    def __call__(self, x: Tensor) -> Tensor:
        x_float = x.float()
        normed = x_float * (x_float.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (normed * (1.0 + self.weight.float())).cast(x.dtype)


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float) -> tuple[Tensor, Tensor]:
    """HF-compatible RoPE: returns (cos, sin) each [1, seq, 1, head_dim]."""
    freqs = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2)[: head_dim // 2] / head_dim))
    positions = Tensor.arange(max_seq_len)
    freqs = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [seq, head_dim//2]
    emb = freqs.cat(freqs, dim=-1)  # [seq, head_dim] — HF doubles freqs
    return emb.cos().reshape(1, max_seq_len, 1, head_dim), emb.sin().reshape(1, max_seq_len, 1, head_dim)


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return (-x2).cat(x1, dim=-1)


def apply_rotary_emb(xq: Tensor, xk: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """HF-style rotate_half RoPE. xq/xk: [B, S, H, D], cos/sin: [1, S, 1, D]."""
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x
    bs, seqlen, n_kv_heads, head_dim = x.shape
    return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

# ============================================================================
# P2: Attention
# ============================================================================

class Gemma3Attention:
    def __init__(self, config: Gemma3Config, layer_idx: int):
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        self.wq = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps)

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # QK norms
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # GQA expand
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # [B, S, H, D] -> [B, H, S, D]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # attention
        attn = (xq @ xk.transpose(-2, -1)) * self.scaling
        attn = attn + mask
        attn = attn.float().softmax(-1).cast(x.dtype)
        out = (attn @ xv).transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.wo(out)

# ============================================================================
# P3: MLP + Decoder Layer
# ============================================================================

class Gemma3MLP:
    def __init__(self, config: Gemma3Config):
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).gelu() * self.up_proj(x))


class Gemma3DecoderLayer:
    def __init__(self, config: Gemma3Config, layer_idx: int):
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        # attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # feedforward block
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        # break fusion to avoid Metal buffer limit (max 31 per kernel)
        return x.contiguous().contiguous_backward()

# ============================================================================
# P4: Full Model
# ============================================================================

class Gemma3:
    def __init__(self, config: Gemma3Config):
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_scale = config.hidden_size ** 0.5

        # precompute RoPE for both local and global (not trainable)
        cos_l, sin_l = precompute_freqs_cis(config.head_dim, config.max_position_embeddings, config.rope_theta_local)
        cos_g, sin_g = precompute_freqs_cis(config.head_dim, config.max_position_embeddings, config.rope_theta_global)
        self.cos_local = cos_l.contiguous().is_param_(False)
        self.sin_local = sin_l.contiguous().is_param_(False)
        self.cos_global = cos_g.contiguous().is_param_(False)
        self.sin_global = sin_g.contiguous().is_param_(False)

    def __call__(self, tokens: Tensor) -> Tensor:
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens) * self.embed_scale

        # build masks
        causal_mask = Tensor.full((1, 1, seqlen, seqlen), float("-inf")).triu(1)
        sliding_mask = Tensor.full((1, 1, seqlen, seqlen), float("-inf")).triu(1)
        # sliding window: also mask beyond window distance
        positions = Tensor.arange(seqlen)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # [S, S], dist[i,j] = j - i
        sliding_mask = sliding_mask + Tensor.where(
            dist < -self.config.sliding_window + 1, Tensor.full((1,), float("-inf")), Tensor.full((1,), 0.0)
        ).reshape(1, 1, seqlen, seqlen)

        for i, layer in enumerate(self.layers):
            is_full = self.config.layer_types[i] == "full_attention"
            cos = self.cos_global[:, :seqlen] if is_full else self.cos_local[:, :seqlen]
            sin = self.sin_global[:, :seqlen] if is_full else self.sin_local[:, :seqlen]
            mask = causal_mask if is_full else sliding_mask
            h = layer(h, cos, sin, mask)

        h = self.norm(h)
        # tied weights: logits = h @ embedding.weight.T
        logits = h @ self.tok_embeddings.weight.T
        return logits

# ============================================================================
# Weight Loading
# ============================================================================

def load_pretrained(model: Gemma3, safetensors_path: str) -> None:
    """Load HF safetensors weights into tinygrad Gemma3 model."""
    from safetensors import safe_open
    import numpy as np

    n_layers = model.config.num_hidden_layers

    # HF name → tinygrad name
    keymap: dict[str, str] = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
    }
    for l in range(n_layers):
        pfx_hf = f"model.layers.{l}"
        pfx_tg = f"layers.{l}"
        keymap.update({
            f"{pfx_hf}.input_layernorm.weight": f"{pfx_tg}.input_layernorm.weight",
            f"{pfx_hf}.post_attention_layernorm.weight": f"{pfx_tg}.post_attention_layernorm.weight",
            f"{pfx_hf}.pre_feedforward_layernorm.weight": f"{pfx_tg}.pre_feedforward_layernorm.weight",
            f"{pfx_hf}.post_feedforward_layernorm.weight": f"{pfx_tg}.post_feedforward_layernorm.weight",
            f"{pfx_hf}.self_attn.q_proj.weight": f"{pfx_tg}.self_attn.wq.weight",
            f"{pfx_hf}.self_attn.k_proj.weight": f"{pfx_tg}.self_attn.wk.weight",
            f"{pfx_hf}.self_attn.v_proj.weight": f"{pfx_tg}.self_attn.wv.weight",
            f"{pfx_hf}.self_attn.o_proj.weight": f"{pfx_tg}.self_attn.wo.weight",
            f"{pfx_hf}.self_attn.q_norm.weight": f"{pfx_tg}.self_attn.q_norm.weight",
            f"{pfx_hf}.self_attn.k_norm.weight": f"{pfx_tg}.self_attn.k_norm.weight",
            f"{pfx_hf}.mlp.gate_proj.weight": f"{pfx_tg}.mlp.gate_proj.weight",
            f"{pfx_hf}.mlp.up_proj.weight": f"{pfx_tg}.mlp.up_proj.weight",
            f"{pfx_hf}.mlp.down_proj.weight": f"{pfx_tg}.mlp.down_proj.weight",
        })

    # load safetensors → numpy → tinygrad
    tg_sd = nn.state.get_state_dict(model)
    with safe_open(safetensors_path, framework="pt") as f:
        hf_keys = set(f.keys())
        mapped_keys = set()
        for hf_key, tg_key in keymap.items():
            assert hf_key in hf_keys, f"missing HF key: {hf_key}"
            assert tg_key in tg_sd, f"missing tinygrad key: {tg_key}"
            # torch bf16 → float32 numpy → tinygrad
            weight = f.get_tensor(hf_key).float().numpy()
            assert weight.shape == tg_sd[tg_key].shape, \
                f"shape mismatch {hf_key}: HF {weight.shape} vs tg {tg_sd[tg_key].shape}"
            tg_sd[tg_key].assign(Tensor(weight))
            mapped_keys.add(hf_key)

        unmapped = hf_keys - mapped_keys
        assert not unmapped, f"unmapped HF keys: {unmapped}"

    # realize all weights
    Tensor.realize(*tg_sd.values())
    print(f"Loaded {len(keymap)} weights from {safetensors_path}")


PRETRAINED_PATH = "model_weights/gemma3-270m/pretrained/model.safetensors"
CKPT_DIR = "model_weights/gemma3-270m/ckpts"


def save_checkpoint(model: Gemma3, path: str) -> None:
    """Save tinygrad model weights to safetensors."""
    from safetensors.numpy import save_file
    sd = nn.state.get_state_dict(model)
    np_sd = {k: v.numpy() for k, v in sd.items() if v.is_param}
    save_file(np_sd, path)
    print(f"Saved {len(np_sd)} weights to {path}")


def load_checkpoint(model: Gemma3, path: str) -> None:
    """Load tinygrad checkpoint from safetensors."""
    from safetensors.numpy import load_file
    np_sd = load_file(path)
    tg_sd = nn.state.get_state_dict(model)
    for k, v in np_sd.items():
        assert k in tg_sd, f"unexpected key: {k}"
        tg_sd[k].assign(Tensor(v))
    Tensor.realize(*tg_sd.values())
    print(f"Loaded {len(np_sd)} weights from {path}")

# ============================================================================
# P5: Training
# ============================================================================

def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Standard next-token prediction loss. Shifts internally."""
    # logits: [B, S, V], targets: [B, S]
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])  # [B*(S-1), V]
    shift_targets = targets[:, 1:].reshape(-1)  # [B*(S-1)]
    return shift_logits.sparse_categorical_crossentropy(shift_targets)


def train_step(model: Gemma3, optimizer: AdamW, tokens: Tensor) -> Tensor:
    logits = model(tokens)
    loss = cross_entropy_loss(logits, tokens)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def train_step_grad_accum(
    model: Gemma3, optimizer: AdamW, microbatches: list[Tensor], accum_steps: int,
) -> float:
    """Gradient accumulation: forward/backward each microbatch, step once."""
    total_loss = 0.0
    for mb in microbatches:
        logits = model(mb)
        loss = cross_entropy_loss(logits, mb) / accum_steps  # scale loss
        loss.backward()
        total_loss += loss.item() * accum_steps  # unscale for logging
    optimizer.step()
    optimizer.zero_grad()
    return total_loss / len(microbatches)


if __name__ == "__main__":
    import os, time

    Tensor.training = True

    config = Gemma3Config()
    model = Gemma3(config)

    # load pretrained if available
    if os.path.exists(PRETRAINED_PATH):
        load_pretrained(model, PRETRAINED_PATH)

    params = nn.state.get_parameters(model)
    optimizer = AdamW(params, lr=1e-5)

    total = sum(p.numel() for p in params)
    print(f"Gemma3 270M — {total:,} params ({total/1e6:.1f}M)")

    # training smoke test
    print("\nTraining (5 steps):")
    for step in range(5):
        tokens = Tensor.randint(1, 64, high=config.vocab_size)
        t0 = time.time()
        loss = train_step(model, optimizer, tokens)
        loss_val = loss.item()
        dt = time.time() - t0
        print(f"  step {step}: loss={loss_val:.4f}  ({dt:.1f}s)")

    # save checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    save_checkpoint(model, os.path.join(CKPT_DIR, "step_5.safetensors"))

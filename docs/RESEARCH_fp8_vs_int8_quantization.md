# FP8 vs INT8 Quantization for LLMs

> Generated: 2026-06-04 | Sources: 7 papers, 6 GitHub issues/PRs, 4 tool docs

## TL;DR

- **FP8** uses floating-point (sign + exponent + mantissa) giving logarithmic spacing; **INT8** uses linear/uniform spacing with an external scale factor
- FP8 beats INT8 for **post-training quantization** of LLMs because its log grid naturally handles activation outliers. INT8 catches up with QAT or advanced methods (SmoothQuant)
- **GGUF/llama.cpp has no FP8 support** because: (a) INT8 block-quants are more accurate per bit, (b) inference is bandwidth-bound not compute-bound, (c) CPUs have zero FP8 SIMD, (d) FP8's per-tensor scaling clashes with GGML's block architecture
- FP8 inference requires **Hopper+ GPUs** (H100/H200) — use vLLM or TensorRT-LLM, not ollama
- For local/CPU inference, GGUF Q8_0 (INT8 block-quant) remains the best 8-bit option

## 1. The Two Formats

### INT8 (Uniform/Linear Quantization)

256 levels uniformly spaced. With symmetric quantization:

```
x_q = round(x / Delta),  Delta = max(|x|) / 127
x_approx = x_q * Delta
```

- Constant step size everywhere
- Good when values are uniformly distributed
- A single outlier at 100x median wastes ~7 bits of precision for all other values

### FP8 (Logarithmic/Floating-Point Quantization)

8 bits split into sign (1) + exponent (3-5) + mantissa (2-3). Two variants:

| Variant | Exponent | Mantissa | Max Value | Use Case |
|---------|----------|----------|-----------|----------|
| **E4M3** | 4 bits | 3 bits | 448 | Forward pass (weights + activations) |
| **E5M2** | 5 bits | 2 bits | 57,344 | Backward pass (gradients) |

- Step size scales with magnitude (constant *relative* precision)
- Dense grid near zero, sparse grid for large values
- E4M3 breaks IEEE 754: no infinity, single NaN (more usable values)

### The Key Mathematical Difference

Between consecutive powers of two [2^a, 2^(a+1)], FP8 has 2^m uniformly spaced values (m = mantissa bits). The step size is 2^(a-m), growing proportionally with magnitude.

**Outlier example** — tensor with values in [-1, 1] and one outlier at 100:
- **INT8**: Delta = 100/127 = 0.787. Values in [-1,1] get ~3 distinct levels. Catastrophic.
- **E4M3**: Values near 0.01 get precision ~0.001. Values near 100 get precision ~8. Bulk precision preserved.

## 2. Why FP8 Beats INT8 for LLM PTQ

Transformer LLMs develop **systematic activation outliers** starting at ~6.7B params (Dettmers et al., 2022):
- ~0.1-1% of channels have magnitudes 10-100x larger than typical values
- Same channels are outliers for every input token
- Removing them causes catastrophic accuracy collapse

**Quantitative impact on INT8** (SmoothQuant paper, OPT models):

| Model | FP16 | INT8 per-tensor PTQ |
|-------|------|---------------------|
| OPT-6.7B | 64.9% | 39.9% |
| OPT-175B | 71.6% | 32.3% |

Per-tensor INT8 produces near-random results for large LLMs.

**FP8 handles this naturally** — its logarithmic grid allocates precision proportional to 1/|x|, so outliers get coarse quantization (acceptable, they're rare) while bulk values retain fine precision.

Kuzmin et al. (Qualcomm, 2022) proved analytically that FP8 has lower MSE than INT8 for Gaussian and heavy-tailed distributions.

## 3. INT8 Methods That Close the Gap

### SmoothQuant (MIT, 2022)
Migrates quantization difficulty from activations to weights via per-channel scaling:
```
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
Y = (X * diag(s)^-1) * (diag(s) * W)
```
- Smooths activation outliers, makes INT8 work
- Zero runtime overhead (fused into preceding LayerNorm)
- OPT-175B: 71.1% accuracy vs 71.6% FP16 (vs 32.3% naive INT8)

### GPTQ (IST Austria, 2022)
Weight-only quantization using Hessian-based error compensation. Primarily W4A16.

### AWQ (MIT, 2023)
Activation-aware weight scaling. Protects salient weight channels identified by activation magnitudes.

### LLM.int8() (Dettmers, 2022)
Mixed-precision decomposition: outlier dims in FP16, rest in INT8. Zero accuracy loss but 15-23% slower.

## 4. What GGUF Q8_0 Actually Does

```c
typedef struct {
    ggml_half d;        // FP16 scale factor
    int8_t  qs[32];     // 32 quantized INT8 values
} block_q8_0;
```

- **Block size 32**, symmetric, zero-point = 0
- Per-block scale: `d = max(|x_0..x_31|) / 127`
- Effective bits/weight: 8.5 (34 bytes per 32 values)
- This is per-group quantization — much better than per-tensor, catches local scale variations

## 5. Why GGUF/Ollama Has No FP8

Four reasons:

1. **INT8 block-quants are more accurate per bit** — INT8 uses all 8 bits for value representation; FP8 splits bits across sign/exponent/mantissa. With GGUF's block scaling (one FP16 scale per 32 values), INT8 achieves better accuracy at the same storage cost.

2. **Bandwidth-bound, not compute-bound** — single-user token generation reads weights from memory sequentially. Compute is idle. FP8's advantage (native tensor core ops) is irrelevant when compute isn't the bottleneck.

3. **No FP8 CPU SIMD** — CPUs have mature INT8 instructions (AVX2, AVX-512 VNNI, ARM NEON). Zero FP8 support on any consumer CPU.

4. **Architectural mismatch** — GGML uses block quantization (scale per 32 elements). FP8 models use per-tensor or per-row scaling. Merging these requires significant GGML changes.

**Status**: Draft PR #10055 adds FP8 types but is blocked by maintainers. NVIDIA collaborator confirmed FP8 support is planned (Apr 2026 discussion #22042).

**Ollama** is a thin wrapper over llama.cpp/GGUF. No FP8 in llama.cpp = no FP8 in Ollama.

## 6. Where to Use FP8

FP8 inference is supported by:

| Tool | FP8 Support | Best For |
|------|------------|----------|
| **vLLM** | W8A8 E4M3, `quantization="fp8"` | Multi-user serving on H100+ |
| **TensorRT-LLM** | W8A8 + FP8 KV cache | Maximum throughput (+144% vs FP16) |
| **Transformer Engine** | Training + inference | FP8 mixed-precision training |

**Hardware requirements**: Hopper (H100) for W8A8; Ada Lovelace (RTX 4090) for limited support; Ampere (A100) and older have no FP8.

## 7. Decision Matrix

| Scenario | Best Choice | Why |
|----------|------------|-----|
| Local inference, CPU/consumer GPU | GGUF Q8_0 or Q4_K_M | Best accuracy/bit, SIMD optimized |
| Maximum local quality at 8-bit | GGUF Q8_0 | INT8 block-quant > FP8 at same budget |
| Multi-user serving, H100 | vLLM FP8 | Compute-bound, native tensor cores |
| Maximum datacenter throughput | TensorRT-LLM FP8 | Fused FP8 kernels, +144% over FP16 |
| Training | Transformer Engine FP8 | Near-BF16 quality at 2x speed |
| Edge/mobile | INT8 (GGUF or TFLite) | Only realistic option |

**The fundamental divide**: GGUF optimizes for **accuracy per bit** (bandwidth-bound). FP8 optimizes for **throughput per dollar** (compute-bound). They serve different use cases.

## Sources

1. Micikevicius et al. "FP8 Formats for Deep Learning." arXiv:2209.05433, Sep 2022.
2. Kuzmin et al. "FP8 Quantization: The Power of the Exponent." arXiv:2208.09225, Aug 2022.
3. Peng et al. "FP8-LM: Training FP8 Large Language Models." arXiv:2310.18313, Oct 2023.
4. Xiao et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs." ICML 2023.
5. Dettmers et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS 2022.
6. Frantar et al. "GPTQ: Accurate Post-Training Quantization for GPTs." ICLR 2023.
7. Lin et al. "AWQ: Activation-aware Weight Quantization." MLSys 2024.
8. llama.cpp PR #10055, Discussion #8780, #22042 — FP8 support requests and status.
9. OCP Microscaling Formats (MX) Specification v1.0, Sep 2023.
10. NVIDIA Transformer Engine docs, vLLM FP8 docs, TensorRT-LLM docs.

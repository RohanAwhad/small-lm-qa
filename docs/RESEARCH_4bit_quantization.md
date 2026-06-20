# 4-Bit Quantization for LLMs & How INT SIMD Does Float Math

> Generated: 2026-06-04 | Sources: 8 papers, community benchmarks, GGML source code

## TL;DR

- **8 major 4-bit methods** exist: GGUF Q4_K_M, GPTQ, AWQ, NF4, NVFP4, MXFP4, IQ4, EXL2
- Quality ranking at 4-bit: EXL2 >= AWQ > NF4 ~ Q4_K_M > GPTQ
- 4-bit 70B **crushes** FP16 8B by ~10 points average. Always pick the bigger quantized model.
- Below 3B params, 4-bit is destructive (>5% accuracy loss)
- Q4_K_M is the community sweet spot: only +2.4% PPL, runs everywhere, 2x faster than FP16
- **INT SIMD trick**: `Y_fp = S_x * S_w * (X_int8 @ W_int8)` — integer GEMM on INT hardware, FP scaling applied once per block

## 1. The INT SIMD Trick: How Integer Hardware Does Float Math

Neural network inference = matrix multiplies: `Y = X @ W`. The quantization trick:

```
X_fp ≈ S_x * X_int8
W_fp ≈ S_w * W_int8

Y_fp = (S_x * X_int8) @ (S_w * W_int8) = S_x * S_w * (X_int8 @ W_int8)
                                           ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
                                           FP scalar    INTEGER GEMM
                                           (1 multiply)  (runs on INT hardware)
```

The integer GEMM runs entirely on integer ALUs. The FP scale multiply happens **once per block** (e.g., per 32 elements), amortizing FP cost.

### GGML's actual Q8_0 dot product:

```c
for (int ib = 0; ib < nb; ++ib) {
    int sumi = 0;
    for (int j = 0; j < 32; j++)
        sumi += x[ib].qs[j] * y[ib].qs[j];   // pure integer
    sumf += sumi * (d_x * d_y);                // FP scale, once per 32 elements
}
```

### CPU SIMD instructions used:

| ISA | Instruction | What it does | Elements/instruction |
|-----|------------|--------------|---------------------|
| **AVX2** | `vpmaddubsw` | 32x uint8*int8 -> 16x int16 (pairwise) | 32 |
| **AVX-512 VNNI** | `vpdpbusd` | 64x uint8*int8 -> 16x int32 (direct) | 64 |
| **ARM NEON** | `SDOT` | 16x int8*int8 -> 4x int32 | 16 |
| **Apple AMX** | undocumented | 32x32 INT8 tile matmul -> INT32 | 1024 |

Key evolution: AVX2 needs 3 instructions (maddubs -> madd -> add) and has INT16 overflow risk. VNNI and SDOT fuse to a single instruction with INT32 accumulators.

### GPU Tensor Cores:

`mma.sync.aligned.m16n8k32.s32.s8.s8.s32` — 8,192 integer ops per warp instruction. INT32 accumulators, FP scaling applied after.

### For Q4 weights (GGUF):

GGML does NOT do native INT4 GEMM. Instead:
1. Unpack 4-bit nibbles to INT8
2. Multiply INT8 weights x INT8 activations using SIMD
3. Apply FP16 block scales after

The Q4_K kernel does ~800 INT ops per 16 FP ops — integer hardware does the heavy lifting.

## 2. The 4-Bit Methods

### Quick Comparison

| Method | Bits/W | How it works | Target | Quality rank |
|--------|--------|-------------|--------|--------------|
| **Q4_K_M** (GGUF) | 4.85 | Block-quant + k-means scales + mixed precision layers | CPU/Apple/GPU | 4th |
| **GPTQ** | ~4.15 | Hessian-based error compensation, column-by-column | NVIDIA GPU | 5th |
| **AWQ** | ~4.13 | Activation-aware channel scaling | NVIDIA/Edge | 2nd |
| **NF4** (QLoRA) | ~4.13 | Non-uniform quantile grid, 16 optimal levels | Training | 3rd |
| **NVFP4** | 4.50 | E2M1 float + E4M3 block scales | Blackwell | N/A (HW) |
| **MXFP4** | 4.25 | E2M1 float + E8M0 shared exponent | Multi-vendor | N/A (HW) |
| **IQ4_XS** (GGUF) | 4.25 | Non-linear LUT + importance matrix | CPU/Apple/GPU | ~Q4_K_M |
| **EXL2** | arbitrary | Per-layer bit allocation via knapsack optimization | NVIDIA GPU | 1st |

### GGUF Q4_K_M internals

256-weight super-blocks, 8 sub-blocks of 32. Per-sub-block 6-bit quantized scales. Mixed precision: half of attention and FFN tensors bumped to Q6_K.

```c
typedef struct {
    ggml_half d, dmin;       // super-block scales (4 bytes)
    uint8_t scales[12];       // 6-bit sub-scales, packed (12 bytes)
    uint8_t qs[128];          // 4-bit values, 2 per byte (128 bytes)
} block_q4_K;                 // 144 bytes per 256 weights = 4.5 bpw
```

### NF4: The 16 magic numbers

Quantile quantization — bin boundaries at standard normal quantiles so each of 16 levels has equal probability mass:

```
{-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
  0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0}
```

Information-theoretically optimal for Gaussian-distributed weights. Verified: only 7.5% of LLaMA-7B neurons are non-normal (expected false-positive rate = 5%).

### AWQ vs GPTQ

| | AWQ | GPTQ |
|--|-----|------|
| Core idea | Protect salient weight channels via scaling | Error compensation via Hessian |
| Calibration sensitivity | Very low | High (can overfit to WikiText) |
| Cross-domain robustness | PPL +0.5-0.6 | PPL +2.3-4.9 |
| Winner at every model size? | Yes | No |

### NVFP4 (Blackwell)

E2M1 format: 8 values per sign = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}. Block size 16, E4M3 FP8 block scales. Hardware dequant is free on Blackwell tensor cores. 4-6x over BF16.

## 3. Benchmarks

### Perplexity at 4-bit (Llama-2-13B, WikiText-2, apples-to-apples)

| Method | Effective BPW | PPL |
|--------|--------------|-----|
| EXL2 4.9 bpw | 4.90 | **4.308** |
| AWQ g32 | ~4.25 | 4.325 |
| Q4_K_M (GGUF) | 4.83 | 4.333 |
| GPTQ g32 act-order | ~4.25 | 4.338 |
| NF4 | ~4.0 | 4.364 |

EXL2 dominates the Pareto frontier. Q4_K_M punches above its weight.

### Speed on Apple Silicon (LLaMA 7B, tok/s text generation)

| Chip | FP16 | Q8_0 | Q4_0 | Q4 speedup |
|------|------|------|------|-----------|
| M1 Max | 23.0 | 40.2 | 61.2 | **2.7x** |
| M2 Max | 24.7 | 41.8 | 66.0 | **2.7x** |
| M4 Max (40c) | 31.6 | 54.1 | 83.1 | **2.6x** |

Text generation is **memory-bandwidth bound** — Q4 is 2x+ faster because half the bytes need loading.

### The "bigger model at lower bits" question

**4-bit 70B vs FP16 8B (Llama-3.1, IJCAI 2025):**

| Benchmark | FP16 8B | 4-bit 70B | Delta |
|-----------|---------|-----------|-------|
| MMLU | 68.1 | 81.4 | **+13.3** |
| GSM8K | 76.7 | 90.8 | **+14.2** |
| Average (5 tasks) | 65.0 | 74.6 | **+9.6** |

**Always pick the bigger quantized model.** Even 70B crushed to IQ2_XXS (~2.1 bpw, 17GB) still beats FP16 8B by 7 points on MMLU.

### Model size threshold for 4-bit

| Size | MMLU drop | Verdict |
|------|-----------|---------|
| <2B | -5 to -10 pts | Destructive |
| 3-4B | -2 to -4 pts | Marginal |
| 7-8B | -1 to -2 pts | Usable |
| 13B+ | <1 pt | Near-lossless |

**Your Gemma3 270M at Q8_0 is fine** — at 270M, even Q8_0 is aggressive. Q4 would likely be unusable at this scale.

## 4. Sub-4-Bit: How Far Can You Go?

| Bits | Method | PPL delta | Viable? |
|------|--------|-----------|---------|
| 3.5 | Q3_K_M | +0.64 | Yes, noticeable but usable |
| 3.0 | EXL2 3.0 | +2.03 | Marginal |
| 2.5 | IQ2_M | varies | Only at 70B+ scale |
| 2.0 | QuIP#, AQLM | varies | Research-grade, 70B+ only |
| <2 | BitNet | N/A | Requires training from scratch |

**Importance matrix (imatrix)** extends usable range by ~0.5 bpw — calibration data identifies which weights matter most. Mandatory below 3 bits.

## 5. Decision Matrix

| Scenario | Best choice |
|----------|------------|
| CPU/Apple Silicon inference | Q4_K_M (GGUF) |
| NVIDIA GPU, single user | EXL2 4.0-5.0 bpw |
| NVIDIA GPU, batched serving | AWQ or GPTQ+Marlin |
| Fine-tuning frozen base | NF4 + QLoRA |
| Blackwell datacenter | NVFP4 |
| Maximum quality at 4-bit | AWQ (inference), NF4+DQ (training) |
| Maximum compression | EXL2 2.5-3.0 bpw |

## Sources

1. Frantar et al. "GPTQ: Accurate Post-Training Quantization for GPTs." ICLR 2023.
2. Lin et al. "AWQ: Activation-aware Weight Quantization." MLSys 2024 Best Paper.
3. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
4. Kawrakow. llama.cpp PR #1684 (K-quants), PR #5590 (IQ4_NL).
5. NVIDIA. "NVFP4: 4-bit Inference on Blackwell." GTC 2025.
6. OCP. "Microscaling Formats (MX) Specification v1.0." Sep 2023.
7. Lee et al. "Comprehensive Study of LLM Quantization." IJCAI 2025.
8. Oobabooga. "Quantization Format Comparison." community benchmarks.
9. Artefact2. "GGUF Quantization Ladder." community benchmarks.
10. GGML source: `ggml-quants.c`, `ggml-common.h` — block structs and dot product kernels.

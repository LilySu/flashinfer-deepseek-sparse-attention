# Batched Pipeline Results — 11.38x Average Speedup

**Date:** 2026-04-11
**128/128 passed, abs_err=0.00 on all workloads**
**Min: 4.16x, Max: 21.76x**

## The Change

Replaced the **per-batch Python loop** (9 kernels × B launches = ~280 launches at B=31) with a **fully batched pipeline** (~10 launches total regardless of B):

1. Single `gather_dequant_fp8` call for all B×max_pages
2. `torch.bmm` for batched GEMM (was per-batch `q[b] @ K.T`)
3. Batched `clamp_`, `mul_`, `sum(dim=1)` 
4. Batched `torch.topk` with seq_len masking
5. Vectorized index remap with `torch.gather`

**Enabled by:** New evaluator (flashinfer-bench 0.1.3.dev67) compares sorted score vectors with atol=0.01 instead of exact I32 index match. torch.bmm, batched sum, batched topk all now accepted.

## Per-Regime Breakdown

| Group | Per-batch (2.99x) | Batched (11.38x) | Improvement |
|-------|-------------------|-------------------|-------------|
| Small (B=1-3) | 5.44x | 4.60x | -0.84x (overhead for B=1) |
| Medium (B=4-8) | 3.30x | **7.19x** | +3.89x |
| Large (B=11-16) | 2.53x | **11.68x** | +9.15x |
| VLarge (B=25-31) | 2.23x | **18.53x** | +16.30x |
| **Overall** | **2.99x** | **11.38x** | **+8.39x** |

## Key Implementation Details

- **Seq_len masking:** Within a workload, batch elements can have variable seq_lens (e.g., seq_len=2 with max_pages=1). Masked with `-inf` before topk.
- **Padding handling:** topk returns indices for masked positions → set to -1 via `topk_tokens[topk_vals == -inf] = -1`.
- **Small-B regression:** B=1 workloads are slightly slower (4.6x vs 5.4x) due to batched op overhead. Could dispatch to per-batch for B <= 2.

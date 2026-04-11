# Gather+Dequant Fusion Results — 2.64x Average Speedup

**Date:** 2026-04-10
**128/128 passed, abs_err=0.00 on all workloads**
**Min: 1.89x, Max: 6.30x**

## Changes from 2.42x baseline

### kernel.cu
- Added `gather_dequant_fp8_kernel` — reads FP8 pages directly from scattered KV cache locations by page index, dequants and writes to contiguous output. Eliminates the intermediate PyTorch gather (`kv_flat_pages[page_indices]`) + separate dequant.

### binding.py
- Replaced `kv_flat_pages[page_indices].reshape(-1)` + `mod.dequant_fp8(selected, K_paged)` with single `mod.gather_dequant_fp8(kv_cache_flat, page_indices, K_paged)`.
- This eliminates 1 kernel launch per batch + 1 intermediate tensor allocation + 1 read/write of the intermediate buffer.

## Cumulative Per-Regime Breakdown

| Group | Baseline (1.70x) | +Index Remap (2.42x) | +Gather+Dequant (2.64x) |
|-------|-------------------|----------------------|-------------------------|
| Small (B=1-3) | 3.36x | 4.62x | 4.83x |
| Medium (B=4-8) | 1.87x | 2.67x | 2.90x |
| Large (B=11-16) | 1.40x | 2.00x | 2.24x |
| VLarge (B=25-31) | 1.23x | 1.77x | **1.97x** |
| **Overall** | **1.70x** | **2.42x** | **2.64x** |

## Current Per-Batch Launch Count: ~8

1. `mod.gather_dequant_fp8(...)` — fused gather + dequant
2. `q_b @ K.T` — cuBLAS GEMM
3. `torch.relu(scores)` — elementwise
4. `scores_relu * w[:, None]` — elementwise
5. `.sum(dim=0)` — reduce
6. `torch.topk(...)` — 2 kernels (gather + radix sort)
7. `mod.fused_index_remap(...)` — fused index remap
8. `topk_indices[b, :k] = remap_out` — copy

At B=31: ~8 × 31 = ~248 launches (down from ~434 at baseline).

## Remaining Opportunities

- Fuse relu + weight_mul (2→1 launches per batch). TVM FFI version had correctness issues in harness — need to investigate if PyTorch-native approach (torch.where, custom autograd) can avoid TVM FFI.
- Fuse remap + copy (already touching the remap output, could write directly to topk_indices if TVM FFI handles storage offsets).
- Stream pipelining — blocked by TVM FFI not respecting PyTorch streams.

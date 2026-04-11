# Fused Index Remap Results — 2.42x Average Speedup

**Date:** 2026-04-10
**128/128 passed, abs_err=0.00 on all workloads**

## Changes Made

### kernel.cu
1. Reverted `dequant_fp8_impl` back to `dequant_fp8_vec_kernel<<<num_pages, 8>>>` (vectorized, 128-bit loads). The wide kernel (128 threads) didn't help per-batch dequant on 10-91 page grids.
2. Added `fused_index_remap_kernel` — replaces 4-5 PyTorch ops per batch:
   - `topk_idx // PAGE_SIZE` (elementwise)
   - `topk_idx % PAGE_SIZE` (elementwise)
   - `page_indices[page_idx_per_token]` (gather)
   - `global_page_idx * PAGE_SIZE + offset_per_token` (elementwise + add)
   - `.to(torch.int32)` (type cast)
   All replaced by a single CUDA kernel that does the entire remap in one launch.

### binding.py
1. Pre-allocated `K_paged_buf` and `topk_remap_buf` before the loop.
2. Replaced the 4-5 index remapping ops with `mod.fused_index_remap(topk_idx, page_indices, remap_out)`.

### What didn't work
- **fused_relu_weight** — produces bit-identical values in isolation but causes INCORRECT_NUMERICAL in the benchmark harness. Root cause unclear — possibly TVM FFI stream ordering or memory allocator interaction with the benchmark's measurement framework. Not worth the risk.
- **Selective bulk dequant** — Python overhead of building set/mapping + GPU gather was worse than the per-batch dequant launches it replaced. Regressed from 1.70x to 1.50x.

## Per-Regime Breakdown

| Group | Before (Step 1) | After (index remap) | Delta |
|-------|-----------------|---------------------|-------|
| Small (B=1-3) | 3.36x | 4.62x | +1.26x |
| Medium (B=4-8) | 1.87x | 2.67x | +0.80x |
| Large (B=11-16) | 1.40x | 2.00x | +0.60x |
| Very Large (B=25-31) | 1.23x | 1.77x | +0.54x |
| **Overall** | **1.70x** | **2.42x** | **+0.72x** |

## Why It Works

At B=31, the per-batch loop previously generated ~14 kernel launches × 31 batches = ~434 launches. The index remap fusion eliminates 5 of those 14, reducing to ~9 × 31 = ~279 launches. That's ~155 fewer launches × ~4us each = **~620us saved from launch overhead alone**, plus the kernel fusion itself avoids intermediate tensor allocation.

## Current Per-Batch Launch Count

1. `kv_flat_pages[page_indices]` — gather
2. `.reshape(-1)` — view (free)
3. `mod.dequant_fp8(...)` — dequant kernel
4. `q_b @ K.T` — cuBLAS GEMM
5. `torch.relu(scores)` — elementwise
6. `scores_relu * w[:, None]` — elementwise
7. `.sum(dim=0)` — reduce
8. `torch.topk(...)` — 2 kernels (gather + radix sort)
9. `mod.fused_index_remap(...)` — single fused kernel
10. `topk_indices[b, :k] = remap_out` — copy

Total: ~10 launches per batch (down from ~14).

## Remaining Optimization Opportunities

With 10 launches × 31 batches = 310 launches, launch overhead is still ~1240us. The biggest remaining targets:
1. **Stream pipelining** — overlap dequant with GEMM across batches
2. **Further launch reduction** — fuse gather + dequant, or reduce PyTorch op overhead
3. **CUDA graphs** — if per-batch shapes can be padded to uniform

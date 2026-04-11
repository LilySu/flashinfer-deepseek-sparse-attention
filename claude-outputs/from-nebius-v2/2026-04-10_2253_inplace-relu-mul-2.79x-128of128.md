# In-Place ReLU+Mul Results — 2.79x Average Speedup

**Date:** 2026-04-10
**128/128 passed, abs_err=0.00 on all workloads**
**Min: 1.93x, Max: 6.78x**

## Change from 2.64x

### binding.py
Replaced:
```python
scores_relu = torch.relu(scores)
w = weights[b]
weighted_scores = scores_relu * w[:, None]
final_scores = weighted_scores.sum(dim=0)
```
With:
```python
scores.clamp_(min=0)
scores.mul_(weights[b][:, None])
final_scores = scores.sum(dim=0)
```

This eliminates 2 intermediate tensor allocations (scores_relu and weighted_scores). Both clamp_ and mul_ are still separate launches but the in-place ops avoid CUDA malloc overhead.

**Why in-place works but TVM FFI didn't:** PyTorch in-place ops run on the same CUDA stream as the preceding matmul (q_b @ K.T). The TVM FFI kernel runs on a different stream (the default CUDA stream), creating a race condition.

## Cumulative Optimization Progression

| Optimization | Small (1-3) | Medium (4-8) | Large (11-16) | VLarge (25-31) | Overall |
|-------------|-------------|--------------|----------------|----------------|---------|
| Baseline | 3.36x | 1.87x | 1.40x | 1.23x | 1.70x |
| +Index Remap | 4.62x | 2.67x | 2.00x | 1.77x | 2.42x |
| +Gather+Dequant | 4.83x | 2.90x | 2.24x | 1.97x | 2.64x |
| **+In-Place ReLU+Mul** | **5.14x** | **3.07x** | **2.33x** | **2.08x** | **2.79x** |

## Current Per-Batch Launch Count: ~7

1. `mod.gather_dequant_fp8(...)` — fused gather + dequant (TVM)
2. `q_b @ K.T` — cuBLAS GEMM
3. `scores.clamp_(min=0)` — in-place ReLU
4. `scores.mul_(w[:, None])` — in-place weight multiply
5. `.sum(dim=0)` — reduce
6. `torch.topk(...)` — 2 kernels (gather + radix sort)
7. `mod.fused_index_remap(...)` — fused index remap (TVM)
8. `topk_indices[b, :k] = remap_out` — copy

~8 launches per batch. At B=31: ~248 launches.

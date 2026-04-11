# Benchmark, Sweep, NCU & Evaluator Results — 2026-04-10

## 1. Full Benchmark: 128/128 passed, 1.64x avg on H200

All workloads pass with abs_err=0.00. Speedups range from 1.15x (large batch) to 4.21x (small batch).

## 2. Threshold Sweep: `//3` wins at 1.68x

| Threshold | Speedup |
|-----------|---------|
| `//3` | **1.68x** (best) |
| `//1` (always bulk) | 1.55x |
| `//2` (current) | 1.53x |
| never bulk | 1.65x |
| 5000 | 1.67x |
| 1000 | 1.67x |
| 3000 | 1.63x |

**Action: Update threshold from `//2` to `//3`** for +0.15x improvement.

## 3. CUDA relu+weight_mul: Blocker

Function never compiled into kernel.cu — the `relu_weight_mul` TVM FFI export doesn't exist. This was approach #12 that already failed. Would need to add it to kernel.cu and re-export.

## 4. Evaluator Check: TOLERANCE FLAGS FOUND

The evaluator appears to have relaxed from exact I32 match to tolerance-based. This could unlock:
- `torch.bmm` batched GEMM (approach #8)
- Batched `.sum(dim=1)` (approach #9)
- Possibly `torch.compile` (approach #13)

## 5. NCU Profiling — H200 Kernel Breakdown (B=4, 4 batches)

| Kernel | % Time | Avg us | Count |
|--------|--------|--------|-------|
| radixSort (topk) | **27.1%** | 23.4 | 4 |
| gatherTopK | **25.3%** | 21.8 | 4 |
| cuBLAS SGEMM | 8.7% | 7.5 | 4 |
| reduce_kernel (sum) | 7.7% | 6.6 | 4 |
| elementwise ops | ~25% | various | many |

**Key insight vs Modal B200 data:** TopK (sort+gather) is now **52.4%** of time (was 22.6% on B200). GEMM dropped from 40.6% to 8.7%. The dequant kernel isn't even showing — hybrid approach eliminated it as a bottleneck.

## Next Steps

1. **Update threshold to `//3`** — easy win, +0.04x over current
2. **Re-test batched approaches** now that tolerance flags exist — could be massive speedup by eliminating per-batch loops
3. **TopK is the #1 bottleneck** at 52% — any way to speed up or approximate topk would have biggest impact
4. **NCU report saved** at `ncu_reports/dsa_profile_20260410_084924.ncu-rep` (94MB) — copy to local before shutdown

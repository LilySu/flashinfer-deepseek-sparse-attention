# opt4 Results: Dequant 128-thread + Pre-alloc — 2026-04-10

## Changes Made

1. **kernel.cu**: Added `dequant_fp8_wide_kernel` with 128 threads/block (1 thread per dim, loop over 64 tokens). Replaced `dequant_fp8_vec_kernel` (8 threads) in `dequant_fp8_impl`.
2. **binding.py**: Pre-allocated `K_paged_buf` before the per-batch loop to avoid `torch.empty` per iteration.

## Results

- **128/128 passed**, abs_err=0.00 on all workloads
- **Average speedup: 1.65x** (vs 1.64x baseline — within noise, no real improvement)
- Max: 4.38x (was 4.21x), Min: 1.15x (unchanged)

## Why It Didn't Help

The dequant kernel occupancy fix (1.6% → higher) should reduce per-launch time from ~33us to ~5-8us. But there are two reasons this didn't translate to wall-clock improvement:

1. **Per-batch dequant operates on small page counts (10-91 pages):** With only 10-91 blocks in the grid, even the 8-thread kernel completes quickly because there's not enough work to expose the low-occupancy problem. The 33us per call was measured with 11923 pages (bulk mode). Per-batch with 50 pages is already fast.

2. **Dequant is NOT on the critical path for per-batch mode:** The per-batch dequant launches for just `num_pages_for_seq` pages. At 50 pages × 33us/launch... wait, it's one launch with 50 blocks, not 50 launches. So the kernel time is already just ~33us regardless of block count (latency-bound). Going to 128 threads doesn't help because the bottleneck was never compute throughput — it was launch overhead and memory latency with too few blocks to hide it.

3. **The real bottleneck is TopK (34%) + launch overhead (28%)** — neither was touched by these changes.

## Key Insight

The NCU profiling showed dequant at 20.8% of kernel time, but:
- Kernel time is only ~72% of wall time
- The remaining 28% is launch overhead
- Dequant's 33us was constant regardless of 8 vs 128 threads because with small grid sizes (10-91 blocks), the SM isn't saturated either way

## What Would Actually Help

1. **Reduce launch count** — CUDA graphs or fused kernels to cut the ~500 launches per call
2. **Fuse the per-batch elementwise chain** — relu + weight_mul + index ops = 6+ launches per batch
3. **TopK optimization** — 34% of compute, can't easily change CUB radix sort
4. **Batch the entire pipeline** (if evaluator tolerance allows) — eliminate the per-batch loop entirely

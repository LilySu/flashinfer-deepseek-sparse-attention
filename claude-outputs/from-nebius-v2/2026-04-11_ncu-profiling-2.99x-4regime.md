# NCU Profiling of 2.99x Solution — 4 Regime Analysis

**Date:** 2026-04-11
**GPU:** NVIDIA H200 (SM90, 132 SMs)
**Solution:** 2.99x average (gather+dequant fusion, index remap fusion, in-place relu+mul)

## Regime D Bottleneck (B=31, 91 pages — the worst case)

| Kernel | Count | Avg (us) | Total (us) | % of kernel time |
|--------|-------|----------|------------|------------------|
| **gather_dequant_fp8** | 29 | 33.2 | 963 | **29.8%** |
| **gatherTopK** | 28 | 29.8 | 836 | **25.8%** |
| **radixSort** | 28 | 23.8 | 667 | **20.6%** |
| SGEMM (cuBLAS) | 29 | 8.6 | 248 | 7.7% |
| reduce (sum) | 28 | 6.6 | 186 | 5.7% |
| elementwise (clamp_) | 28 | 3.9 | 108 | 3.3% |
| elementwise (mul_) | 29 | 2.8 | 82 | 2.5% |
| copy (topk_indices) | 28 | 2.6 | 72 | 2.2% |
| fused_index_remap | 28 | 2.4 | 68 | 2.1% |
| **TOTAL kernel time** | | | **3236 us** | |

**Estimated wall time:** 3236us kernel + ~248 launches × 4us = **~4228us** (4.23ms)
**Reference latency (B=31):** ~5ms → speedup ~1.18x (matches observed ~2.0-2.2x for VLarge)

Note: NCU counts are ~28-29 instead of 31 because launch-skip/count doesn't capture all batches perfectly.

## Key Findings

### 1. TopK is the #1 bottleneck (46.4% combined)
- gatherTopK (25.8%) + radixSort (20.6%) = **46.4% of kernel time**
- 29.8us + 23.8us = **53.6us per batch**
- At B=31: ~1503us — nearly half of all kernel time
- CUB radix sort is a black box — can't optimize without replacing it

### 2. gather_dequant is #2 (29.8%)
- Still 33us per launch despite fusion — the fusion eliminated the *gather launch*, but the dequant kernel itself is unchanged
- 8 threads/block, 91 blocks per grid → 728 threads total on 132 SMs
- Occupancy is still low, but improving it didn't help (tested: wide kernel was same speed)
- This is latency-bound: 33us is the memory access time for 91 pages regardless of thread count

### 3. The "small" ops add up (13.8%)
- clamp_ (3.3%) + mul_ (2.5%) + copy (2.2%) + index_remap (2.1%) + reduce (5.7%) = 15.8%
- These are 5 launches per batch × ~3us avg = ~15us per batch
- At B=31: ~465us total

### 4. GEMM is cheap (7.7%)
- 8.6us per call, cuBLAS is efficient for these small matrices
- Not worth optimizing

## Comparison: Old (1.66x) vs New (2.99x) at B=31

| Category | Old (us) | New (us) | Reduction |
|----------|----------|----------|-----------|
| Dequant | 1056 (20.8%) | 963 (29.8%) | -9% (fusion saved gather, kernel same speed) |
| TopK | 1725 (33.9%) | 1503 (46.4%) | -13% (unchanged, but now dominant) |
| GEMM | 274 (5.4%) | 248 (7.7%) | -9% |
| Elementwise | 1813 (35.6%) | 330 (10.2%) | **-82%** (fusions + in-place) |
| Reduce | 213 (4.2%) | 186 (5.7%) | -13% |
| Index remap | (in elementwise) | 68 (2.1%) | **fused from ~300us** |
| **Total kernel** | **5086** | **3236** | **-36%** |
| **Launches** | ~1000 | ~248 | **-75%** |
| **Launch overhead** | ~4000us | ~992us | **-75%** |

## What's Left to Optimize (Ranked by Impact)

### Tier 1: TopK (46.4% = 1503us at B=31)
- gatherTopK scales with seq_len (8us at 640 tokens → 30us at 5824)
- radixSort is constant ~24us per call
- **Only option:** Replace torch.topk with a custom kernel that produces identical indices
- Risk: CUB radix sort has specific tie-breaking — matching it exactly is hard

### Tier 2: Launch Overhead (~992us, 19% of wall time)
- 248 launches × 4us = ~992us
- **CUDA graphs** could eliminate this entirely
- Challenge: variable shapes per batch (seq_len differs)

### Tier 3: gather_dequant (29.8% = 963us)
- 33us per launch × 29 = 963us
- Kernel is latency-bound at 8 threads/block with 91 blocks
- **Potential:** Multi-page-per-block kernel (process 2-4 pages per block, more threads)
- Or: async memcpy + compute overlap within the kernel

### Tier 4: Small ops (clamp_ + mul_ + copy = 8.0% = 260us)
- Already optimized with in-place ops
- Could fuse clamp_ + mul_ into single kernel (saves ~80us from 1 fewer launch × 31)

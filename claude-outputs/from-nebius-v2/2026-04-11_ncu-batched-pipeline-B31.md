# NCU Profile: Batched Pipeline at B=31

**Date:** 2026-04-11
**Total kernel time: 257.3us (0.257ms) + ~80us launch overhead (20 launches) = ~337us**
**vs per-batch: 3236us kernel + 1032us overhead = ~4268us**
**Kernel time reduction: 12.6× fewer microseconds**

## Chronological Kernel Listing

| # | Operation | Grid | Block | Time | Compute% | Memory% | Occ% | % Total |
|---|-----------|------|-------|------|----------|---------|------|---------|
| 0 | int64 cast (block_table→long) | (62,1,1) | (128,1,1) | 3.4us | 2.2% | 154.9% | 6.2% | 1.3% |
| 1 | copy (block_table reshape) | (6,1,1) | (128,1,1) | 4.3us | 0.1% | 10.4% | 6.1% | 1.7% |
| 2 | scatter_gather (page_indices) | (62,1,1) | (128,1,1) | 7.1us | 1.9% | 81.9% | 6.2% | 2.8% |
| 3 | int64 arithmetic | (62,1,1) | (128,1,1) | 2.5us | 0.8% | 208.6% | 6.1% | 1.0% |
| 4 | int64 add | (62,1,1) | (128,1,1) | 2.6us | 0.8% | 389.7% | 6.1% | 1.0% |
| 5 | copy (reshape) | (124,1,1) | (128,1,1) | 3.2us | 3.1% | 169.0% | 6.2% | 1.2% |
| 6 | float arithmetic | (62,1,1) | (128,1,1) | 2.3us | 0.7% | 112.9% | 6.1% | 0.9% |
| 7 | masked_fill (-inf) | (62,1,1) | (128,1,1) | 2.2us | 0.6% | 146.7% | 5.9% | 0.9% |
| 8 | fill_(-1) topk_indices | (62,1,1) | (128,1,1) | 1.8us | 0.7% | 2.3% | 6.1% | 0.7% |
| 9 | **convert_q_fp8→fp32** | (992,1,1) | (256,1,1) | 2.5us | 10.4% | 101.5% | **71.4%** | 1.0% |
| 10 | fill_(-1) topk_indices | (62,1,1) | (128,1,1) | 1.8us | 0.6% | 2.3% | 6.1% | 0.7% |
| 11 | copy | (6,1,1) | (128,1,1) | 4.1us | 0.1% | 11.0% | 6.1% | 1.6% |
| 12 | **gather_dequant_fp8** | **(2821,1,1)** | **(8,1,1)** | **44.1us** | **36.3%** | 1.6% | **31.0%** | **17.1%** |
| 13 | **torch.bmm (cuBLAS)** | **(1,91,31)** | **(128,1,1)** | **91.5us** | **76.6%** | 1.4% | **23.8%** | **35.6%** |
| 14 | **clamp_ (ReLU)** | (11284,1,1) | (128,1,1) | **22.3us** | 20.6% | 3.0% | **80.3%** | **8.7%** |
| 15 | **mul_ (weights)** | (45136,1,1) | (128,1,1) | **33.6us** | 57.7% | 2.0% | **80.5%** | **13.1%** |
| 16 | **sum(dim=1)** | (1411,1,1) | (32,4,1) | **18.4us** | 14.9% | 2.7% | **47.8%** | **7.1%** |
| 17 | arange (mask positions) | (91,1,1) | (64,1,1) | 2.0us | 0.5% | 1.4% | 3.1% | 0.8% |
| 18 | topk (comparison) | (353,1,1) | (128,1,1) | 4.7us | 11.5% | 24.1% | 16.6% | 1.8% |
| 19 | masked_fill (topk -inf→-1) | (177,1,1) | (128,1,1) | 2.9us | 1.8% | 315.7% | 7.8% | 1.1% |
| | **TOTAL** | | | **257.3us** | | | | |

## Key Findings

### 1. torch.bmm is the new #1 bottleneck (35.6%, 91.5us)

- Grid: **(1, 91, 31)** — cuBLAS dispatches 91×31 = 2821 thread blocks
- **Compute throughput: 76.6%** — actually compute-bound! This is the highest compute utilization of any kernel
- Occupancy: 23.8% — decent for a GEMM
- This is ONE launch for all 31 batches vs 31 × 8.6us = 267us in per-batch path
- **3× faster than per-batch approach** (91.5us vs 267us)

### 2. gather_dequant is #2 (17.1%, 44.1us)

- Grid: **(2821,1,1)** = 31 × 91 pages, Block: (8,1,1) — same 8-thread kernel
- **Compute: 36.3%, Occupancy: 31.0%** — much better than per-batch (1.6% occupancy)!
- With 2821 blocks, the GPU actually has enough work to hide memory latency
- 44.1us for all 31 batches vs 31 × 33us = 1023us per-batch = **23× faster**

### 3. mul_ (weights) is #3 (13.1%, 33.6us)

- Grid: **(45136,1,1)** = huge grid, 80.5% occupancy
- 57.7% compute throughput — actually doing real work
- This is `scores.mul_(weights[:B, :, None])` on [31, 64, 5824] = 11.5M elements

### 4. clamp_ (ReLU) is #4 (8.7%, 22.3us)

- Grid: (11284,1,1), 80.3% occupancy
- 20.6% compute — simple operation but large tensor

### 5. sum(dim=1) is #5 (7.1%, 18.4us)

- Grid: (1411,1,1), 47.8% occupancy
- Reduction across 64 heads for 31 × 5824 elements

### 6. TopK is now TINY (1.8%, 4.7us)

- Was 46.4% in per-batch (1503us for 31 batches)
- Now just 4.7us for the batched topk on [31, 5824]
- **321× faster** — batched topk uses many SMs vs single-SM per-batch

### 7. Setup ops (0-11) total ~35us (13.6%)

- Block table reshape, int64 casts, copies, fills
- 12 small kernel launches before the main pipeline
- These could potentially be eliminated

## Bottleneck Summary

| Category | Time (us) | % | Per-batch equivalent |
|----------|-----------|---|---------------------|
| **GEMM (bmm)** | **91.5** | **35.6%** | 267us (3.0× faster) |
| **Dequant** | **44.1** | **17.1%** | 1023us (23× faster) |
| **mul_ (weight)** | **33.6** | **13.1%** | 108us (3.2× faster) |
| **clamp_ (relu)** | **22.3** | **8.7%** | 82us (3.7× faster) |
| **sum(dim=1)** | **18.4** | **7.1%** | 186us (10× faster) |
| **topk** | **4.7** | **1.8%** | 1503us (320× faster) |
| Setup ops | 35.3 | 13.7% | — |
| Misc (remap etc) | 7.4 | 2.9% | 140us |
| **TOTAL kernel** | **257.3** | | **3236us (12.6× less)** |
| **Launch overhead** | **~80** | | **~1032us (12.9× less)** |
| **Est wall time** | **~337** | | **~4268us** |

## Where the Next 2× Is

1. **GEMM (91.5us, 35.6%)** — already at 76.6% compute utilization. Hard to optimize further without changing the algorithm.

2. **Dequant (44.1us, 17.1%)** — 8 threads/block with 2821 blocks. Now that the grid is large, occupancy IS relevant. The wide kernel (128 threads) might actually help now, unlike per-batch where grid was only 91.

3. **Setup ops (35us, 13.6%)** — 12 kernels for block_table manipulation. Could be eliminated by pre-processing block_table in a single kernel.

4. **clamp_ + mul_ (55.9us, 21.8%)** — two separate kernels on the same large tensor. Fusing would save 1 launch + 1 full memory pass = ~20us.

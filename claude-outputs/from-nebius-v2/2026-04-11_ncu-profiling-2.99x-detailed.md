# NCU Profiling — 2.99x Solution, Detailed Analysis

**Date:** 2026-04-11
**GPU:** NVIDIA H200 (SM 9.0, 132 SMs, 1.98 GHz SM, 3.20 GHz DRAM)
**NCU reports:** `ncu_reports/dsa_v3_{A,B,C,D}_20260411_011158.ncu-rep`

---

## Single Batch Iteration — Data Flow (Regime D: B=31, 91 pages, seq=5824)

Each batch element executes this exact 9-kernel pipeline:

| Step | Operation | DSA Stage | Grid | Block | Time | Compute% | Memory% | Occupancy | Notes |
|------|-----------|-----------|------|-------|------|----------|---------|-----------|-------|
| 1 | `gather_dequant_fp8` | K page dequant | (91,1,1) | (8,1,1) | **33.2us** | 1.5% | 23.3% | **1.6%** | 91 pages × 8 threads = 728 active threads. 8448 bytes/page read, 32768 bytes/page written. Latency-bound: 80%+ stalled on memory. |
| 2 | `cuBLAS SGEMM` | Q×K^T matmul | (2,182,1) | (64,1,1) | **8.4us** | 31.7% | 360.2% | 8.5% | [64,128] × [128,5824]^T → [64,5824]. Memory throughput >100% means L2 cache reuse. Small matrix = low SM utilization. |
| 3 | `clamp_(min=0)` | ReLU activation | (364,1,1) | (128,1,1) | **2.8us** | 5.9% | 544.9% | 15.4% | In-place on [64,5824] = 1.49M elements. Bandwidth-bound (reads + writes same tensor). |
| 4 | `mul_(w[:,None])` | Head weighting | (1456,1,1) | (128,1,1) | **3.9us** | 16.3% | 388.8% | 56.5% | In-place multiply [64,5824] × [64,1] broadcast. Highest occupancy of all custom ops. |
| 5 | `sum(dim=0)` | Head reduction | (46,1,1) | (32,4,1) | **6.9us** | 1.2% | 227.1% | 5.2% | ATen reduce: [64,5824] → [5824]. 46 blocks × 128 threads. Reduction across 64 heads. |
| 6 | `gatherTopK` | Top-K selection | **(1,1,1)** | (1024,1,1) | **27.4us** | 0.4% | 1.5% | 49.9% | **Single block!** Scans 5824 elements to find top-2048. 1024 threads but grid=1 means only 1 SM is used. |
| 7 | `radixSort` | Top-K sort | **(1,1,1)** | (64,1,1) | **23.4us** | 0.1% | 4.6% | 3.1% | CUB radix sort on 2048 elements. **Single block, 64 threads, 3.1% occupancy.** Sorts the top-K indices by value. |
| 8 | `fused_index_remap` | Index remapping | (8,1,1) | (256,1,1) | **2.4us** | 0.2% | 8.4% | 12.1% | Maps topk indices → global page tokens. 2048 elements, tiny. |
| 9 | `copy_to_output` | Write output | (1,1,1) | (128,1,1) | **2.6us** | 0.0% | 12.9% | 5.5% | Copy 2048 int32 values to topk_indices[b]. |
| | **BATCH TOTAL** | | | | **110.9us** | | | | |

**Per-batch breakdown:** gather_dequant (30.0%) + TopK (45.8%) + GEMM (7.6%) + elementwise (8.6%) + reduce (6.2%) + remap+copy (4.5%)

---

## Scaling Across Regimes

### Per-Kernel Average Time (us)

| Kernel | A (B=1, 10pg) | B (B=4, 50pg) | C (B=16, 70pg) | D (B=31, 91pg) | Scaling |
|--------|---------------|---------------|-----------------|-----------------|---------|
| gather_dequant | 32.0 | 32.9 | 33.1 | 33.2 | ~Constant (latency-bound) |
| cuBLAS SGEMM | 7.0 | 7.5 | 8.3 | 8.6 | Slight ↑ with seq_len |
| clamp_(ReLU) | 2.5 | 2.6 | 2.7 | 2.8 | ~Constant |
| mul_(weight) | 3.3 | 3.6 | 3.6 | 3.9 | ~Constant |
| sum(dim=0) | 6.4 | 6.6 | 6.7 | 6.9 | ~Constant |
| **topk_gather** | **8.1** | **22.6** | **26.8** | **29.8** | **Scales with seq_len (3.7×)** |
| topk_radixSort | 22.7 | 23.6 | 23.8 | 23.8 | ~Constant (always sorts 2048) |
| fused_index_remap | 2.5 | 2.4 | 2.4 | 2.4 | Constant |
| copy_to_output | 2.8 | 2.6 | 2.5 | 2.6 | Constant |
| **Per-batch total** | **87.3** | **104.4** | **110.1** | **113.3** | |

### Total Time Breakdown by Regime

| Component | A (B=1) | B (B=4) | C (B=16) | D (B=31) |
|-----------|---------|---------|----------|----------|
| Kernel time | 138us | 465us | 1655us | 3236us |
| Launches | 18 | 42 | 138 | 258 |
| Launch overhead (~4us) | ~72us | ~168us | ~552us | ~1032us |
| **Est. wall time** | **~210us** | **~633us** | **~2207us** | **~4268us** |
| Reference latency | ~1020us | ~1500us | ~4700us | ~5100us |
| **Est. speedup** | **~4.9x** | **~2.4x** | **~2.1x** | **~1.2x** |

---

## Critical Observations

### 1. TopK is the #1 bottleneck (46% of kernel time at B=31)

- **gatherTopK:** grid=(1,1,1) — runs on **a single SM**. With 1024 threads, occupancy is 49.9% on that one SM, but **131 other SMs are idle**. At seq_len=5824, it takes 29.8us per call.
- **radixSort:** grid=(1,1,1) — also **single SM**, 64 threads, **3.1% occupancy**. CUB's radix sort for 2048 elements. Fixed ~23.8us regardless of input size.
- Combined: **53.6us per batch** = 1503us at B=31 = **46.4%** of kernel time.
- These kernels have the highest per-call cost and the worst utilization (single-SM execution).

### 2. gather_dequant is constant at 33us (30% of kernel time)

- 91 blocks × 8 threads = 728 threads across 132 SMs = **5.5 threads per SM**
- Occupancy: **1.6%** — the SM is essentially idle between memory transactions
- Compute throughput: 1.5% — almost no math is done
- Memory throughput: 23.3% — reading 8448 bytes/page × 91 pages = ~751KB, writing 32KB/page × 91 = ~2.9MB
- This is **pure memory latency** — the kernel waits for DRAM, doesn't have enough threads/warps to hide it
- The wide kernel (128 threads) didn't help because 91 blocks still only uses 91/132 SMs

### 3. clamp_ + mul_ are cheap but add up to 2 launches per batch

- clamp_: 2.8us, 15.4% occupancy, bandwidth-bound
- mul_: 3.9us, 56.5% occupancy, bandwidth-bound
- Combined: 6.7us kernel + 8us launch overhead = **14.7us per batch**
- At B=31: **~456us total** (kernel + overhead)
- Fusing into 1 kernel would save ~4us overhead × 31 = 124us

### 4. sum(dim=0) has poor occupancy (5.2%)

- 46 blocks × 128 threads = 5888 threads. For 132 SMs, that's ~45 threads/SM
- But each block only uses (32,4,1) = 128 threads doing a tree reduction across 64 heads
- 6.9us per call, scales linearly with B

### 5. copy_to_output is unnecessary overhead

- 2.6us per call just to copy 2048 int32 values
- This is the `topk_indices[b, :actual_topk] = remap_out` — but we already write directly via fused_index_remap
- **Wait — this shouldn't exist.** The current code writes directly. NCU may be capturing a different iteration.

### 6. Launch overhead is 24% of estimated wall time

- 258 launches × ~4us = ~1032us
- Wall time ~4268us → **24.2% is pure launch overhead**
- CUDA graphs would eliminate this entirely

---

## Where the Time Goes (Regime D, B=31)

```
┌─────────────────────────────────────────────────────────────────┐
│                    EST. WALL TIME: ~4268us                      │
├──────────────────────────────────┬──────────────────────────────┤
│     KERNEL TIME: 3236us (76%)    │  LAUNCH OVERHEAD: 1032us    │
├──────────────────────────────────┤         (24%)               │
│ topk_gather     836us (19.6%)    │                              │
│ gather_dequant  963us (22.6%)    │  258 launches × ~4us each    │
│ topk_radixSort  667us (15.6%)    │                              │
│ cuBLAS SGEMM    248us (5.8%)     │  CUDA graphs could save     │
│ sum(dim=0)      186us (4.4%)     │  ~800-1000us here           │
│ mul_(weight)    108us (2.5%)     │                              │
│ clamp_(ReLU)     82us (1.9%)     │                              │
│ copy_to_output   72us (1.7%)     │                              │
│ fused_index_remap 68us (1.6%)    │                              │
│ setup (q_conv+fill) 6us (0.1%)   │                              │
└──────────────────────────────────┴──────────────────────────────┘
```

---

## Actionable Optimization Targets

| Priority | Target | Current Cost | Potential Savings | Approach |
|----------|--------|-------------|-------------------|----------|
| **1** | Launch overhead | 1032us (24%) | ~800us | CUDA graphs |
| **2** | topk_gather | 836us (20%) | ? | Custom gather kernel (grid>1) |
| **3** | gather_dequant occupancy | 963us (23%) | ~500us if 10us/call | Multi-page per block, more warps |
| **4** | clamp_ + mul_ fusion | 190us + 248us overhead | ~124us | Single CUDA kernel |
| **5** | copy_to_output elimination | 72us + 124us overhead | ~196us | Verify not needed |

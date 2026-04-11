# Deep NCU Profiling — 4 Regime Analysis on H200

**Date:** 2026-04-10
**GPU:** NVIDIA H200 (SM90, 132 SMs)
**Real contest params:** num_pages=11923, page_size=64, head_dim=128, num_heads=64, topk=2048

## Regime Definitions

| Regime | B | Seq Len | Pages/Seq | Total Pages Used | Bulk Dequant (//2) | Bulk Dequant (//3) | Contest Speedup |
|--------|---|---------|-----------|-----------------|-------------------|-------------------|-----------------|
| A | 1 | 640 | 10 | 10 | No | No | ~4.2x |
| B | 4 | 3,200 | 50 | 200 | No | No | ~2.1x |
| C | 16 | 4,480 | 70 | 1,120 | No | No | ~1.2x |
| D | 31 | 5,824 | 91 | 2,821 | No | No | ~1.15x |

Note: With //3 threshold (3974), none of these regimes trigger bulk dequant. With //2 (5961), only D would barely not trigger either. The per-batch dequant path dominates.

## Category Rollup Across Regimes

| Category | Regime A | Regime B | Regime C | Regime D |
|----------|----------|----------|----------|----------|
| **GEMM** | 42us (2.9%) | 180us (4.8%) | 266us (5.4%) | 274us (5.4%) |
| **TopK (sort+gather)** | 185us (12.6%) | 1084us (28.7%) | 1603us (32.5%) | 1725us (33.9%) |
| **Reduce (sum)** | 42us (2.8%) | 159us (4.2%) | 212us (4.3%) | 213us (4.2%) |
| **Dequant** | 196us (13.3%) | 790us (20.9%) | 1058us (21.5%) | 1056us (20.8%) |
| **Elementwise** | 991us (67.5%) | 1552us (41.1%) | 1788us (36.3%) | 1813us (35.6%) |
| **Total kernel time** | 1469us | 3778us | 4931us | 5086us |
| **Launches** | 123 | 393 | 500* | 500* |
| **Est. launch overhead** | 492us (25%) | 1572us (29%) | 2000us (29%) | 2000us (28%) |

*Regime C/D capped at 500 launch capture limit — actual launches for B=16 would be ~513, B=31 would be ~999.

## Critical Findings

### 1. Elementwise Ops Are the #1 Time Sink (35-67%)

This was invisible in the shallow profile. The elementwise category includes:
- `distribution_elementwise_grid_stride_kernel` — **~235us fixed cost** (4 launches, ~59us each) — this is PyTorch's random number generation for the test setup, runs once regardless of B. In real contest this may be different.
- `float8_copy_kernel` — **~113us fixed cost** (2 launches, ~57us each) — Q FP8 copy
- `elementwise_kernel (direct_copy)` — **~208us fixed cost** (2 launches, ~104us each) — large copy operation
- `launch_clamp_scalar` — **scales with B** (B+2 launches, ~8-25us each) — relu clamp operation
- Various index arithmetic — `BUnaryFunctor`, `AUnaryFunctor`, `CUDAFunctor_add` — all ~2-3us each but B launches each

**Key insight:** ~556us of elementwise is FIXED OVERHEAD (setup/copy), not per-batch work. This explains why small-B gets 4x — the per-batch work is tiny relative to the baseline overhead that the reference implementation also pays.

### 2. TopK Is the Scaling Bottleneck (13% → 34% as B grows)

| | Regime A (B=1) | Regime D (B=31) | Per-call |
|---|---|---|---|
| gatherTopK | 51us (3.4%) | 973us (19.1%) | 8→30us |
| radixSort | 134us (9.1%) | 752us (14.8%) | 22→24us |

- **gatherTopK scales with seq_len** — 8.4us at 640 tokens → 30.4us at 5824 tokens (3.6x for 9x more tokens)
- **radixSort is ~constant** per call (~23us) — CUB radix sort on 2048 elements
- Combined: 30+24 = **54us per batch** at large seq_len

### 3. Dequant Is Constant at ~33us Per Call (1.6% Occupancy!)

`dequant_fp8_vec_kernel`: 32-33us per call regardless of regime. With only 8 threads per block and `num_pages` blocks, occupancy is **1.6%** — the SM is nearly idle.

- Memory throughput: 0.1-0.5% — barely touching DRAM
- Compute throughput: 0.2-1.5% — barely computing
- **This kernel is entirely latency-bound** — it's waiting on memory with almost no parallelism

### 4. GEMM Is Actually Cheap (~7-9us per call, 5.4%)

cuBLAS SGEMM: only 7-9us per call. At B=31 that's ~274us total. With 4.3→8.5% occupancy it's not saturating the GPU but it's fast enough. Not the bottleneck.

### 5. Occupancy Problems (Kernel by Kernel)

| Kernel | Occupancy | Diagnosis |
|--------|-----------|-----------|
| dequant_fp8_vec | **1.6%** | 8 threads/block, single block per SM wave |
| radixSort | **3.1%** | CUB internal, can't control |
| SGEMM | 3-9% | Small matrix (64x128 × 128×seq), cuBLAS overhead |
| reduce_kernel | **5.4%** | sum(dim=0) on 64 rows |
| gatherTopK | 31-50% | Reasonable |
| elementwise (relu/clamp) | 12-17% | Small tensors |
| copy kernels | 45-86% | Only ones running efficiently |

### 6. Launch Overhead is ~28% of Wall Time

At ~4us per kernel launch on H200, the per-batch loop creates massive launch overhead:
- B=1: 123 launches → 492us overhead (25%)
- B=31: ~999 launches → ~4000us overhead (28%)
- **Each batch iteration launches ~10 kernels** via PyTorch ops

## Optimization Opportunities (Ranked by Impact)

### Tier 1: High Impact

**A. Fuse per-batch elementwise chain (relu + weight_mul + index arithmetic)**
- Currently: relu (clamp) + weight_mul (elementwise) + index ops = ~6-8 kernel launches per batch at ~2-5us each
- Potential: Single CUDA kernel, eliminate ~6 launches per batch
- Saves: ~24us kernel time + ~24us launch overhead per batch
- At B=31: ~1500us saved → could be 1.4x → 1.6x on large-B workloads

**B. Fix dequant occupancy (1.6% → 50%+)**
- Current: 8 threads per block, 1 page per block → only 8 threads active per SM
- Fix: Use 256+ threads per block, tile multiple tokens per thread, process multiple pages per block
- Saves: Could cut 33us → ~5us per call (6x), saving ~800us at B=31

**C. CUDA graph or stream-based launch amortization**
- Capture the per-batch loop as a CUDA graph (if shapes are uniform)
- Or: use multiple streams to overlap kernel launches
- Saves: Could cut launch overhead from 28% to <5%

### Tier 2: Medium Impact

**D. Batched TopK (if tolerance allows)**
- Currently: B separate topk calls (sort + gather = ~54us each)
- If evaluator tolerance allows approximate topk: could use batched CUB sort
- Saves: up to 1000us at B=31

**E. Batched GEMM (if tolerance allows)**
- Currently: B separate matmul calls (~8us each)
- If evaluator tolerance allows: torch.bmm could batch all B
- Saves: ~250us at B=31, but also eliminates B-1 launch overheads

### Tier 3: Lower Impact

**F. Batched sum (if tolerance allows)** — saves ~200us at B=31
**G. Persistent kernel** — single kernel launch for entire pipeline per batch

## What the Shallow Profile Missed

1. **Elementwise was invisible** — the shallow run only captured 50 kernels with launch-skip, missing the fixed-cost setup kernels entirely
2. **Scaling behavior** — without multi-regime, you couldn't see that TopK grows from 13% to 34% while GEMM stays flat at 5%
3. **Dequant occupancy crisis** — 1.6% occupancy wasn't surfaced without full metrics
4. **Launch overhead** — the shallow run skipped the first 20 launches, hiding the true launch count
5. **Fixed vs per-batch costs** — couldn't distinguish ~556us fixed overhead from scaling costs

## Files

- `ncu_reports/dsa_deep_A_*.ncu-rep` — Regime A (B=1)
- `ncu_reports/dsa_deep_B_*.ncu-rep` — Regime B (B=4)
- `ncu_reports/dsa_deep_C_*.ncu-rep` — Regime C (B=16)
- `ncu_reports/dsa_deep_D_*.ncu-rep` — Regime D (B=31)

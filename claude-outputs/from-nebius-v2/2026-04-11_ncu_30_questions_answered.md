# 30 NCU Questions Answered — v3 Profiles (2.99x Solution)

**Profiles:** dsa_v3_{A,B,C,D}_20260411_011158.ncu-rep, H200 (SM90, 132 SMs)
**Context:** New DsaTopkIndexerEvaluator compares sorted score vectors (atol=0.01, rtol=0.01). Exact I32 match NO LONGER required.

---

## I. Kernel Inventory & Time Attribution

### Q1: Complete kernel launch sequence (Profile D, 2 batch iterations)

```
  0:   33.18μs  grid=   91  blk=   8  occ=  1.6%  gather_dequant
  1:    8.45μs  grid=  364  blk=  64  occ=  8.5%  GEMM
  2:    2.75μs  grid=  364  blk= 128  occ= 15.4%  clamp_ (ReLU)
  3:    3.87μs  grid= 1456  blk= 128  occ= 56.5%  mul_ (weight)
  4:    6.88μs  grid=   46  blk= 128  occ=  5.2%  sum(dim=0)
  5:   27.39μs  grid=    1  blk=1024  occ= 49.9%  TopK-gather
  6:   23.39μs  grid=    1  blk=  64  occ=  3.1%  TopK-sort
  7:    2.40μs  grid=    8  blk= 256  occ= 12.1%  IndexRemap
  8:    2.62μs  grid=    1  blk= 128  occ=  5.5%  copy_to_output ←batch 1 end
  9:   33.44μs  grid=   91  blk=   8  occ=  1.6%  gather_dequant
  ... (identical 9-kernel pattern repeats for each batch element)
```

9 kernels per batch element, perfectly regular, zero variation in order or grid topology.

### Q2: Launch count & overhead per pipeline pass

| Profile | B | Launches/pass | Compute (us) | Overhead (us) | Overhead % | Wall (us) |
|---------|---|--------------|-------------|--------------|-----------|----------|
| A | 1 | 18 | 138 | 72 | 34% | 210 |
| B | 4 | 34 | 372 | 134 | 27% | 507 |
| C | 16 | 138 | 1,655 | 552 | 25% | 2,207 |
| D | 31 | 258 | 3,236 | 1,032 | 24% | 4,268 |

Launch overhead is 24-34% of wall-clock at all batch sizes. Under the new evaluator, batching to ~8 total launches reduces overhead to 32us (< 1%).

### Q3: Custom code vs framework internals (Profile D)

| Category | Time (us) | % | Launches | You Control? |
|----------|----------|---|----------|-------------|
| Custom CUDA (gather_dequant, IndexRemap, Q-convert) | 1,033 | 31.9% | 58 | YES |
| ATen elementwise (clamp_, mul_, copy, sum) | 452 | 14.0% | 115 | Partially (can batch) |
| Library black-box (cuBLAS GEMM, CUB TopK) | 1,751 | 54.1% | 85 | **WAS NO → NOW YES via batching** |

Under the new evaluator, the "black-box" 54.1% is now batchable via torch.bmm and batched topk.

### Q4: Init vs pipeline kernels

`distribution_elementwise_grid_stride_kernel` at ~224us runs once (tensor init). Not captured in v3 profiles due to launch-skip. All 258 captured actions are pipeline kernels.

### Q5: Per-batch vs global kernels

| Kernel | A (B=1) | B (B=4) | C (B=16) | D (B=31) | Per-B | Type |
|--------|---------|---------|----------|----------|-------|------|
| gather_dequant | 1 | 5 | 15 | 29 | 1.0 | PER-BATCH |
| GEMM | 1 | 5 | 15 | 29 | 1.0 | PER-BATCH |
| TopK-gather | 2 | 4 | 15 | 28 | 1.0 | PER-BATCH |
| TopK-sort | 2 | 4 | 15 | 28 | 1.0 | PER-BATCH |
| Reduce(sum) | 2 | 4 | 15 | 28 | 1.0 | PER-BATCH |
| IndexRemap | 2 | 4 | 15 | 28 | 1.0 | PER-BATCH |
| clamp/mul(vec) | 4 | 7 | 17 | 31 | ~2.0 | PER-BATCH (2 launches) |
| mul_(elem) | 2 | 4 | 15 | 28 | 1.0 | PER-BATCH |
| copy(unroll) | 1 | 4 | 15 | 28 | 1.0 | PER-BATCH |
| Q-convert | 1 | 1 | 1 | 1 | — | GLOBAL |

EVERY kernel except Q-convert is per-batch. With batching, all become 1 call total.

### Q6: % of kernel time by category at each batch size

| Kernel | A (B=1) | B (B=4) | C (B=16) | D (B=31) | Trend |
|--------|---------|---------|----------|----------|-------|
| gather_dequant | 23.2% | 35.3% | 30.0% | 29.8% | STABLE |
| TopK-gather | 11.8% | 19.4% | 24.3% | 25.8% | GROWS |
| TopK-sort | 33.0% | 20.2% | 21.6% | 20.6% | SHRINKS |
| GEMM | 5.1% | 8.0% | 7.5% | 7.7% | GROWS |
| Reduce(sum) | 9.2% | 5.6% | 6.0% | 5.7% | SHRINKS |
| clamp/mul(vec) | 5.6% | 3.6% | 2.7% | 2.7% | SHRINKS |
| mul_(elem) | 4.9% | 3.1% | 3.3% | 3.3% | SHRINKS |
| IndexRemap | 3.7% | 2.0% | 2.2% | 2.1% | SHRINKS |
| copy(unroll) | 2.0% | 2.2% | 2.3% | 2.2% | STABLE |
| **TopK COMBINED** | **44.8%** | **39.6%** | **45.8%** | **46.4%** | **DOMINANT** |

---

## II. Scaling Behavior

### Q7: Per-launch duration scaling

| Kernel | A (B=1,10p) | D (B=31,91p) | D/A | Scales With |
|--------|-------------|-------------|-----|-------------|
| gather_dequant | 32.00 us | 33.20 us | 1.0x | Constant |
| TopK-gather | 8.11 us | 29.85 us | **3.7x** | Context length |
| TopK-sort | 22.72 us | 23.82 us | 1.0x | Constant |
| GEMM | 7.04 us | 8.55 us | 1.2x | ~Constant |
| Reduce(sum) | 6.35 us | 6.64 us | 1.0x | Constant |
| IndexRemap | 2.53 us | 2.43 us | 1.0x | Constant |

**TopK-gather is the ONLY kernel with per-launch scaling** — 3.7x from 640 to 5824 tokens. Under batching, this becomes one call processing [B, seq_len] which may use a more efficient CUB dispatch.

### Q8: Per-pipeline-pass time (avg × B)

| Kernel | A (B=1) | B (B=4) | C (B=16) | D (B=31) |
|--------|---------|---------|----------|----------|
| gather_dequant | 32 us | 132 us | 530 us | 1,029 us |
| TopK-gather | 8 us | 90 us | 429 us | 925 us |
| TopK-sort | 23 us | 94 us | 381 us | 739 us |
| GEMM | 7 us | 30 us | 133 us | 265 us |
| Reduce(sum) | 6 us | 26 us | 107 us | 206 us |
| **TopK COMBINED** | **31 us** | **184 us** | **809 us** | **1,664 us** |

### Q9: Which batch regime dominates the contest score?

| Regime | B range | Workloads | % of 128 | Current avg | Impact weight |
|--------|---------|-----------|----------|-------------|---------------|
| Small | 1-3 | 12 | 9% | 5.44x | Low |
| Medium | 4-8 | 46 | 36% | 3.30x | Medium |
| Large | 11-16 | 33 | 26% | 2.53x | High |
| VLarge | 25-31 | 37 | 29% | 2.23x | Highest |

Large + VLarge = 70 workloads (55%). Optimizing B=31 has 3x more impact than B=1.

### Q10: Bottleneck crossover

| Profile | Dequant % | TopK % | GEMM % | Dominant |
|---------|----------|--------|--------|----------|
| A (B=1) | 23.2% | 44.8% | 5.1% | TopK |
| B (B=4) | 35.3% | 39.6% | 8.0% | TopK |
| C (B=16) | 30.0% | 45.8% | 7.5% | TopK |
| D (B=31) | 29.8% | 46.4% | 7.7% | TopK |

TopK dominates at ALL batch sizes. No crossover — it's consistently 40-46%. Under batching, TopK will likely shrink significantly (batched CUB is more efficient) and dequant or GEMM may become dominant.

### Q11: Launch overhead growth

Linear growth: 18 → 34 → 138 → 258 launches. Overhead fraction DECREASES (34% → 24%) because kernel time grows faster than launch count. Under batching, overhead drops to ~32us regardless of B.

---

## III. GPU Utilization & Occupancy

### Q12-13: Grid size, SM utilization, occupancy (Profile D)

| Kernel | Grid | Block | Warps | SMs Used | Occ% | Limiter | Full Warp? |
|--------|------|-------|-------|----------|------|---------|-----------|
| gather_dequant | 91 | **8** | 23 | 69% | **1.6%** | warps | **NO (8 threads!)** |
| TopK-gather | **1** | 1024 | 32 | **1%** | 49.9% | warps | Yes |
| TopK-sort | **1** | 64 | 2 | **1%** | 3.1% | regs | Yes |
| GEMM | 364 | 64 | 728 | 100% | 8.5% | regs | Yes |
| Reduce(sum) | 46 | 128 | 184 | 35% | 5.2% | regs | Yes |
| IndexRemap | 8 | 256 | 64 | 6% | 12.1% | warps | Yes |
| clamp_ | 364 | 128 | 1,456 | 100% | 15.4% | warps | Yes |
| mul_ | 1,456 | 128 | 5,824 | 100% | 56.5% | warps | Yes |
| copy | 1 | 128 | 4 | 1% | 5.5% | warps | Yes |

### Q14: Waves per SM

| Kernel | Waves/SM | Assessment |
|--------|----------|-----------|
| gather_dequant | 0.02 | Extremely underloaded |
| TopK-gather | 0.00 | Single SM |
| TopK-sort | 0.00 | Single SM |
| GEMM | 0.23 | Underloaded |
| Reduce(sum) | 0.04 | Underloaded |
| mul_ | 0.69 | Acceptable |
| clamp_ | 0.17 | Underloaded |

### Q15: Is low occupancy the actual problem?

For **gather_dequant**: YES and NO. 1.6% occupancy with 80% long_scoreboard = memory-latency-bound with no warps to hide latency. But increasing to 128 threads didn't help because the grid is only 91 blocks — the kernel finishes in one wave regardless. **Batching to grid=2821** changes this: 2821 blocks on 132 SMs = 21 blocks/SM, enough work to benefit from higher occupancy.

For **TopK**: NO. Grid=1 is the fundamental constraint. No occupancy fix can help when the algorithm is single-block.

### Q16: Single-block kernels

- TopK-gather: grid=1, block=1024, 32 warps on 1 SM. 131 SMs idle. 25.8% of kernel time.
- TopK-sort: grid=1, block=64, 2 warps on 1 SM. 131 SMs idle. 20.6% of kernel time.
- copy: grid=1, block=128. Shouldn't exist (phantom copy).
- **Combined: 48.5% of kernel time on 0.76% of the GPU.**

Under batching, torch.topk on [31, 5824] may dispatch 31 independent CUB sorts, potentially using 31 SMs.

### Q17: Non-warp-aligned blocks

**gather_dequant: 8 threads per block** — 0.25 warps. 24 threads in each warp are inactive. 75% of warp scheduler capacity wasted. This is the worst launch configuration in the entire pipeline.

---

## IV. Memory System

### Q18: Sector efficiency

| Kernel | Theoretical Sectors | Ideal Sectors | Efficiency | Assessment |
|--------|-------------------|---------------|-----------|-----------|
| gather_dequant | 215,579 | 122,395 | **1.76x** | UNCOALESCED — 76% wasted |
| TopK-gather | 9,553 | 9,409 | 1.02x | OK |
| TopK-sort | 1,536 | 1,536 | 1.00x | Perfect |
| GEMM | 419,328 | 419,328 | 1.00x | Perfect |
| Reduce(sum) | 49,504 | 47,320 | 1.05x | OK |
| IndexRemap | — | — | **1.84x** | UNCOALESCED |

gather_dequant's 1.76x comes from 8 threads writing float4 at 64-byte stride. Under batching, the kernel itself doesn't change, but more concurrent blocks may amortize the inefficiency.

### Q19: L2 hit rate degradation

| Kernel | A (B=1) | D (B=31) | Trend |
|--------|---------|----------|-------|
| gather_dequant | 81.4% | 81.7% | Stable |
| GEMM | 76.1% | 79.3% | Stable |
| Reduce(sum) | 57.6% | **27.4%** | **DEGRADES** |
| TopK-gather | 75.5% | 71.2% | Stable |

Reduce(sum) L2 hit rate drops 2x from B=1 to B=31. Under batching with .sum(dim=1), the reduction pattern changes entirely — may improve or worsen depending on tensor layout.

### Q20: L1 hit rate

| Kernel | L1 Hit% | Implication |
|--------|---------|------------|
| TopK-gather | **86.8%** | Excellent locality, data fits in L1. Slow due to grid=1, not memory |
| gather_dequant | 67.2% | Good — FP8 pages have spatial locality within a page |
| TopK-sort | 50.0% | Moderate — CUB radix sort has mixed access patterns |
| GEMM | 26.4% | Low — streaming through K matrix, not reused |
| Reduce(sum) | 4.4% | Very low — reduction writes back without reuse |

### Q21: DRAM bytes per launch

| Kernel | DRAM Read (KB) | Assessment |
|--------|---------------|-----------|
| GEMM | 2,972 | Largest — reading K matrix |
| Reduce(sum) | 1,526 | Second — reading scores + writing output |
| gather_dequant | 756 | FP8 pages (91 pages × ~8KB each) |
| TopK-sort | 105 | Small |
| TopK-gather | 41 | Small |

### Q22: Does working set fit in L2?

L2 cache ≈ 50MB on H200. Logits tensor at B=31: 31 × 5824 × 4 bytes = 722KB. **Easily fits.** L2 pressure is NOT the cause of slowdown. The Reduce(sum) L2 degradation is from the reduction output, not input pressure.

---

## V. Warp Execution & Stalls

### Q23-24: Warp stall reasons and diagnosis (Profile D)

| Kernel | #1 Stall | #2 | #3 | Diagnosis |
|--------|----------|----|----|-----------|
| gather_dequant | long_scoreboard 80% | selected 11% | wait 8% | **MEMORY-LATENCY BOUND** — needs more warps |
| TopK-gather | not_selected 23% | barrier 19% | math_pipe 18% | Warp scheduling pressure on 1 SM |
| TopK-sort | wait 28% | no_instructions 28% | selected 21% | Synchronization overhead in CUB |
| GEMM | selected 32% | short_scoreboard 18% | long_scoreboard 14% | **HEALTHY** — actively computing |
| Reduce(sum) | no_instructions 39% | long_scoreboard 35% | wait 8% | Instruction-starved + memory-latency |
| IndexRemap | long_scoreboard 64% | no_instructions 19% | short_scoreboard 8% | **MEMORY-LATENCY BOUND** — random lookup |

### Q25: Warp latency ranking

```
  IndexRemap          :   61.1 cyc/inst  ██████████████████████████████
  clamp/mul(vec)      :   39.6 cyc/inst  ███████████████████
  mul_(elem)          :   26.9 cyc/inst  █████████████
  copy(unroll)        :   26.3 cyc/inst  █████████████
  Reduce(sum)         :   19.9 cyc/inst  █████████
  TopK-gather         :   13.5 cyc/inst  ██████
  gather_dequant      :    9.7 cyc/inst  ████
  TopK-sort           :    4.9 cyc/inst  ██
  GEMM                :    3.2 cyc/inst  █
```

IndexRemap at 61.1 cyc/inst is the worst — random page_indices lookup. Under batching, this becomes a vectorized operation over [B, 2048] which should be much more efficient.

### Q26: Branch divergence

| Kernel | Divergent Branches | Impact |
|--------|-------------------|--------|
| TopK-gather | **500** | ±11% duration variance (23.6-37.0us), data-dependent |
| Reduce(sum) | 4 | Negligible |
| All others | 0 | None |

TopK-gather is the ONLY kernel with meaningful divergence. Under batching, torch.topk on [B, seq_len] handles each batch independently — divergence still exists but amortized across B.

### Q27: Stall profile change A→D

| Kernel | B=1 | B=31 | Change |
|--------|-----|------|--------|
| gather_dequant | long_scoreboard=79%, wait=11% | long_scoreboard=80%, selected=11% | Stable — always memory-bound |
| GEMM | selected=40%, long_scoreboard=24% | selected=32%, short_scoreboard=18% | Shifts from pure compute to memory pressure |
| TopK-gather | wait=33%, barrier=25% | not_selected=23%, barrier=19% | More warp competition at larger context |

GEMM transitions from compute-dominated at B=1 to mixed compute+memory at B=31 as the K matrix grows. Under batched bmm, the larger matrix may push GEMM further into memory-bound territory.

---

## VI. Strategic (Under New Evaluator)

### Q28: Untouchable vs optimizable

**Under EXACT MATCH (old evaluator):**
- Untouchable: 54.1% (cuBLAS GEMM + CUB TopK + ATen sum = 1,751us)
- Optimizable: 45.9% (custom CUDA + ATen elementwise)
- Ceiling: ~3.5-4.0x

**Under NEW EVALUATOR (score comparison, atol=0.01):**
- Untouchable: **0%** — everything can be batched
- Now batchable: 59.9% (1,937us — TopK + GEMM + sum were locked, now free)
- Already optimizable: 31.9% (1,031us — gather_dequant + IndexRemap)
- Trivial: 8.2% (266us — in-place ops)
- Ceiling: **~8-10x** (eliminate per-batch loop entirely)

### Q29: Highest ROI optimization

| Option | Approach | Speedup | Engineering | ROI |
|--------|----------|---------|-------------|-----|
| **A** | **Batch everything** (torch.bmm + batched ops) | **5-8x** | 2-4 hours | **BEST** |
| B | Batch dequant only + kill phantom copy | ~3.3x | 1 hour | Moderate |
| C | CUDA graphs (TVM FFI rewrite) | ~3.7x | 4-8 hours | Poor |

**Option A projected time budget (B=31):**

| Component | Current | Batched | Savings |
|-----------|---------|---------|---------|
| Launches | 280 | ~8 | 97% |
| Launch overhead | 1,032 us | 32 us | 97% |
| GEMM | 265 us | 15-30 us | 89% |
| TopK | 1,664 us | 200-400 us | 76% |
| Sum | 206 us | 10-20 us | 93% |
| Dequant | 1,029 us | 50-100 us | 90% |
| Elementwise | 208 us | 20-40 us | 85% |
| **Wall-clock** | **~4,268 us** | **~400-750 us** | **82-91%** |

### Q30: What to profile next (after batched pipeline)

1. Does torch.bmm on [31,64,128] × [31,128,5824] saturate compute or memory bandwidth?
2. Does batched torch.topk on [31, 5824] use multiple SMs or still grid=1 per batch?
3. Is batched dequant (grid=2821) now bandwidth-limited instead of latency-limited?
4. What's the new launch count and is the overhead now < 5% of wall-clock?
5. Did the bottleneck shift from TopK to dequant or GEMM?
6. Is there further fusion opportunity in the batched pipeline (e.g., GEMM + relu + weight_mul epilogue)?
7. Does torch.compile work now and provide additional gains on the batched pipeline?

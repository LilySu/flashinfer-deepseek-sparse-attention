# Large-Batch Optimization Plan — DSA Indexer

**Date:** 2026-04-10
**Focus:** Batch sizes B >= 11 (70 of 128 workloads, currently dragging average to 1.57x)
**Goal:** Lift large-batch speedup from 1.07-1.41x → 1.5x+ to push overall average past 1.66x

---

## Current Performance Snapshot

| Group | B range | Count | Avg speedup | Min | Max |
|-------|---------|-------|-------------|-----|-----|
| Small | 1-3 | 12 | 3.10x | 2.39x | 4.22x |
| Medium | 4-8 | 46 | 1.73x | 1.50x | 2.09x |
| Large | 11-16 | 33 | 1.29x | 1.26x | 1.41x |
| Very Large | 25-31 | 37 | 1.13x | 1.07x | 1.17x |

Large + Very Large = **70 workloads (55%)** averaging **1.20x**. These dominate the overall average.

## Where Wall Time Goes (Large Batch, B=31 from NCU)

| Component | Time (us) | % of wall | Per-batch? | Launches |
|-----------|-----------|-----------|------------|----------|
| TopK (sort+gather) | 1725 | 24% | Yes (~54us/batch) | 2 × B |
| Elementwise (relu, mul, index) | 1813 | 25% | Mostly yes | ~6-8 × B |
| Dequant FP8→FP32 | 1056 | 15% | Yes (~34us/batch) | 1 × B |
| Launch overhead | ~4000 | 28% | Yes (~4us × ~1000) | ~1000 total |
| GEMM (Q @ K.T) | 274 | 4% | Yes (~9us/batch) | 1 × B |
| Reduce (sum) | 213 | 3% | Yes (~7us/batch) | 1 × B |
| Fixed setup | ~556 | 8% | No | ~8 |

**Key insight:** At B=31, there are ~1000 kernel launches. Launch overhead alone (~4ms) nearly equals the reference latency (~5ms). The per-batch Python loop creates ~10 CUDA kernel launches per batch element via separate PyTorch ops.

## Batch Size Partitioning (Revised)

Based on the data, the workloads naturally cluster into these regimes:

| Regime | B | max_pages | Workload count | Dominant bottleneck |
|--------|---|-----------|----------------|---------------------|
| Small | 1-3 | 1-8 | 12 | Fixed overhead (already fast) |
| Medium | 4-8 | 1-45 | 46 | Dequant + elementwise |
| Large | 11-16 | 1-91 | 33 | Launch overhead + TopK + elementwise |
| Very Large | 25-31 | 1-91 | 37 | Launch overhead dominates, ~1000 launches |

The crossover from "medium" to "large" is around B=11 where total launches exceed ~350 and launch overhead starts dominating.

---

## 10 Approaches for Large-Batch Optimization

Each approach includes a unit test specification that targets large-batch workloads specifically.

---

### Approach 1: Fused Post-GEMM Kernel (ReLU + Weight-Multiply + Sum)

**What:** Replace the per-batch chain of `torch.relu(scores)` → `scores * weights` → `.sum(dim=0)` with a single CUDA kernel that reads scores, applies ReLU, multiplies by weights, and reduces across heads in one pass.

**Why:** Currently these are 4-6 separate kernel launches per batch (~2-5us each kernel + 4us launch overhead each). At B=31, this is ~31 × 6 = 186 launches, ~750us kernel time + ~750us launch overhead = ~1500us total.

**Expected savings:** Eliminate ~5 launches per batch → ~155 fewer launches at B=31 → ~620us saved from launch overhead alone + kernel fusion savings.

**Unit test:**
```python
# Test: fused_post_gemm at B=30, max_pages=91
# Input: scores [64, seq_len], weights [64]
# Expected: identical final_scores as relu(scores) * w[:, None]).sum(dim=0)
# Measure: wall-clock time of fused vs unfused for 100 iterations
```

---

### Approach 2: CUDA Graphs for the Per-Batch Loop

**What:** Capture the entire per-batch iteration (dequant → GEMM → relu → weight → sum → topk → index remap) as a CUDA graph. Replay it B times with different input pointers.

**Why:** CUDA graphs eliminate all CPU→GPU launch overhead. At B=31 with ~10 launches per batch, we have ~310 launches in the loop. At ~4us per launch, that's ~1240us of pure launch overhead.

**Caveat:** Graph capture requires fixed tensor shapes. The per-batch seq_len varies, but max_pages varies per batch element — need to pad or use the max shape and mask.

**Expected savings:** ~1000-1500us at B=31 if shapes can be fixed.

**Unit test:**
```python
# Test: graph_capture at B=30, uniform seq_len (all batches same pages)
# Compare: graph replay vs eager loop, verify bit-identical outputs
# Measure: latency reduction from graph amortization
```

---

### Approach 3: Batched GEMM with Per-Batch Accumulation Guard

**What:** Instead of B separate `q[b] @ K.T` calls, batch all Q and K into padded tensors and use `torch.bmm`. BUT — the CLAUDE.md warns that batched GEMM changes accumulation order and breaks correctness.

**Investigation needed:** Determine if `torch.bmm` with identical shapes produces bit-identical results to sequential `matmul`. If shapes are padded to the same size, cuBLAS should use the same algorithm.

**Why:** Eliminates B-1 GEMM launches (~8us each) + B-1 launch overheads.

**Expected savings:** ~400us at B=31 if correctness holds.

**Unit test:**
```python
# Test: batched_gemm_correctness at B=30, max_pages=91
# Compare: sequential q[b] @ K[b].T vs torch.bmm(Q_padded, K_padded.transpose(-1,-2))
# Check: bit-identical results (abs_err must be exactly 0.00)
# If not bit-identical: measure max absolute error, determine if within tolerance
```

---

### Approach 4: Vectorized Dequant Kernel Selection (Wide for Small Grid, Vec for Large)

**What:** Use the wide kernel (128 threads) for per-batch dequant with small page counts, but switch back to the vectorized kernel (8 threads with uint4 loads) for bulk dequant or large page counts. The vectorized kernel has better memory bandwidth utilization through 128-bit loads.

**Why:** The current results show the wide kernel helped small-B (per-batch with few pages) but the overall average dropped. For large-B workloads, the vectorized kernel's coalesced 128-bit loads may be more efficient when there are enough blocks to saturate the SMs.

**Expected savings:** Recover the 1.66x baseline for large-B while keeping the 4x gains for small-B.

**Unit test:**
```python
# Test: kernel_selection at page_counts=[10, 50, 91, 500, 11923]
# Compare: wide kernel vs vec kernel latency at each page count
# Determine: crossover point where vec kernel becomes faster
```

---

### Approach 5: Persistent Fused Kernel (Dequant + GEMM + Post-Processing)

**What:** Write a single CUDA kernel that, for one batch element, does: load FP8 pages → dequant → multiply by Q → ReLU → weight → partial sum. This eliminates ALL intermediate tensors and launches for the dequant-to-sum pipeline.

**Why:** Each batch element currently requires: 1 dequant launch + 1 GEMM launch + 1 ReLU + 1 weight-mul + 1 sum = 5+ launches. A persistent kernel reduces this to 1 launch per batch.

**Complexity:** High — need to implement matrix multiply in CUDA (or use CUTLASS). But the matrices are small enough (64×128 × 128×seq_len) that a simple tiled implementation may suffice.

**Expected savings:** ~4 fewer launches per batch × 31 batches × 4us = ~500us from launch reduction + fusion gains.

**Unit test:**
```python
# Test: fused_dequant_gemm at B=1 first, then B=30
# Input: raw FP8 pages + Q tensor
# Expected: identical scores as dequant-then-matmul path
# Measure: single-kernel time vs multi-kernel pipeline
```

---

### Approach 6: Reduce Per-Batch Python Overhead (Minimize Torch Ops)

**What:** Audit every PyTorch operation in the per-batch loop and minimize intermediate tensor creation. Specifically:
- Pre-compute `page_indices` for all batches before the loop
- Replace `torch.topk` with a CUDA kernel that writes directly to `topk_indices`
- Replace `page_idx_per_token // PAGE_SIZE` and `% PAGE_SIZE` with a fused index-remap kernel
- Use in-place operations where possible

**Why:** Even without CUDA graphs, reducing the number of PyTorch ops reduces both launch count and Python-side overhead. Each torch op has ~5-10us of Python dispatch overhead on top of the CUDA launch.

**Expected savings:** ~3-4 fewer kernel launches per batch at B=31 → ~400-500us total.

**Unit test:**
```python
# Test: minimal_loop at B=30, max_pages=91
# Compare: current loop vs optimized loop, verify identical topk_indices
# Measure: total loop time (Python wall clock), CUDA kernel count via profiler
```

---

### Approach 7: Stream Pipelining (Overlap Dequant with GEMM)

**What:** Use 2 CUDA streams to overlap batch[i]'s GEMM+post-processing with batch[i+1]'s dequant. While the GPU is computing scores for one batch, it's simultaneously dequantizing pages for the next batch.

**Why:** Dequant is memory-bound. GEMM is compute-bound. They use different GPU resources and can overlap. Currently they execute sequentially on the default stream.

**Expected savings:** At B=31 with ~34us dequant and ~9us GEMM, we can hide most dequant time behind GEMM: ~30 × 34us = ~1000us of dequant hidden.

**Unit test:**
```python
# Test: stream_pipeline at B=30, max_pages=91
# Setup: 2 streams, pre-allocate double-buffered K_paged
# Compare: single-stream vs pipelined, verify identical outputs
# Measure: wall-clock reduction, verify overlap via nsys trace
```

---

### Approach 8: Bulk Dequant Threshold Tuning + Selective Bulk

**What:** The current threshold (`total_pages_used > num_pages_total // 3 = 3974`) means NO workload triggers bulk mode (max B*pages = 2730). Investigate:
- Lower the threshold so large-B workloads use bulk dequant
- Or: use "selective bulk" — dequant only the unique pages actually referenced in `block_table` for all B batches at once, rather than per-batch

**Why:** At B=30 with max_pages=91, we're making 30 separate dequant calls for potentially overlapping page sets. If many batches share pages, a single bulk dequant of the union would be cheaper.

**Expected savings:** Replace 30 dequant launches with 1 → save ~29 × 4us launch overhead = ~116us + eliminate redundant page dequants.

**Unit test:**
```python
# Test: bulk_threshold at B=30, max_pages=91
# Measure: page overlap ratio (unique pages used / total page references)
# Compare: per-batch dequant vs bulk-all vs selective-bulk
# Verify: correctness for each strategy
```

---

### Approach 9: Custom TopK Kernel (Replace CUB Radix Sort)

**What:** The current `torch.topk` uses CUB radix sort which launches 2 kernels (gatherTopK + radixSort) totaling ~54us per batch at B=31. For k=2048 out of seq_len=5824, a partial sort or heap-based selection could be faster.

**Why:** TopK is 34% of kernel time at B=31 (1725us). A custom kernel doing approximate or exact top-2048 selection in a single launch could cut this significantly.

**Caveat:** CLAUDE.md says topk is "UNTOUCHABLE" — but that's about changing the *call*, not replacing it with a faster equivalent. If results are bit-identical, a custom kernel should be valid.

**Expected savings:** If we can do topk in ~20us instead of ~54us, that's 34us × 31 = ~1050us saved.

**Unit test:**
```python
# Test: custom_topk at seq_len=[640, 2048, 4096, 5824]
# Input: random float32 scores
# Expected: same indices as torch.topk (exact match)
# Measure: custom kernel time vs torch.topk time
```

---

### Approach 10: Pre-sorted Page Indices + Early Termination

**What:** Before the per-batch loop, pre-compute which pages have the highest aggregate potential across heads. Use this to sort pages by expected importance and potentially skip dequanting low-importance pages.

**Why:** At B=30 with 91 pages per seq, we dequant all 91 pages but only need the top 2048/64 ≈ 32 pages worth of tokens. If we can cheaply estimate page importance (e.g., from Q norms), we could dequant only ~40-50 pages instead of 91.

**Caveat:** This is approximate and may break correctness if the "unimportant" pages contain tokens that should be in the top-2048. Would need a verification pass.

**Expected savings:** ~50% fewer pages to dequant per batch → ~500us at B=31.

**Unit test:**
```python
# Test: page_importance_estimation at B=30, max_pages=91
# Measure: what fraction of top-2048 tokens come from the top-50% of pages
# If >99%: early termination is viable
# If <95%: too risky for correctness
```

---

## Recommended Priority Order

| Priority | Approach | Expected Impact | Difficulty | Risk |
|----------|----------|----------------|------------|------|
| **1** | **#4: Kernel selection** | Recover 1.66x baseline | Low | None |
| **2** | **#8: Bulk threshold tuning** | Reduce launch count | Low | None |
| **3** | **#1: Fused post-GEMM** | -1500us at B=31 | Medium | None |
| **4** | **#6: Reduce Python overhead** | -400-500us at B=31 | Medium | None |
| **5** | **#7: Stream pipelining** | -1000us at B=31 | Medium | Low |
| **6** | **#2: CUDA graphs** | -1000-1500us at B=31 | High | Medium (shape constraints) |
| **7** | **#3: Batched GEMM** | -400us at B=31 | Low | High (correctness) |
| **8** | **#5: Persistent fused kernel** | Best possible perf | Very High | Medium |
| **9** | **#9: Custom TopK** | -1050us at B=31 | High | Medium (correctness) |
| **10** | **#10: Page importance** | -500us at B=31 | Medium | High (correctness) |

## Unit Test Framework

All tests should:
1. Use real contest data (load from `mlsys26-contest/` workloads with B >= 25)
2. Compare against the current implementation for bit-identical results
3. Report wall-clock time (100 iterations, 3 warmup)
4. Report CUDA kernel launch count (via `torch.cuda.nvtx` or manual counting)
5. Be runnable standalone: `python3 test_approach_N.py`

## Next Steps

1. First implement Approach #4 (kernel selection) to recover the 1.66x baseline
2. Then implement Approach #8 (bulk threshold tuning) as it's low-risk
3. Then tackle Approach #1 (fused post-GEMM) for the biggest single improvement
4. Combine winning approaches and re-run full 128-workload benchmark

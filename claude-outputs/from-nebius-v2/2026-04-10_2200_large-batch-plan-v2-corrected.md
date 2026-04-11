# Large-Batch Optimization Plan v2 (Corrected)

**Date:** 2026-04-10
**Focus:** B >= 11 (70 of 128 workloads, currently 1.20x avg)

## Corrections from v1

**DELETED — proven correctness failures:**
- ~~#3 Batched GEMM~~ — torch.bmm selects different cuBLAS algo, already tested = INCORRECT_NUMERICAL
- ~~#5 Persistent fused kernel~~ — can't reimplement cuBLAS SGEMM with bit-identical accumulation
- ~~#9 Custom TopK~~ — can't match CUB radix sort bit-for-bit tie-breaking
- ~~#10 Page pruning~~ — skipping pages breaks exact top-2048 match by definition

**CORRECTED — #1 Fused post-GEMM:**
- ReLU + weight_mul: SAFE to fuse (elementwise, order-independent)
- .sum(dim=0): UNSAFE to fuse (ATen tree reduction, non-associative FP, different order = different topk)
- Corrected scope: fuse relu + weight_mul ONLY, leave .sum(dim=0) as torch call

## Implementation Steps

### Step 1: Recover Baseline (revert wide kernel)
- Switch `dequant_fp8_impl` back to `dequant_fp8_vec_kernel<<<num_pages, 8>>>`
- Per-batch dequant on 10-91 pages is latency-bound regardless of thread count
- 8 threads with uint4 128-bit loads has better instruction efficiency
- Expected: recover 1.66x average

### Step 2: Selective Bulk Dequant (BIGGEST WIN)
- Before loop: compute union of all unique page indices across all B batches
- Single dequant launch for unique pages only (~91 pages max, not all 11923)
- Build page_id → local_index mapping for per-batch lookup
- Replaces B dequant launches with 1 launch
- At B=31: eliminates ~30 launches + ~30 × 4us = 120us launch overhead + removes redundant dequants for shared pages
- Zero correctness risk — same pages, same kernel, same values

### Step 3: Fuse ReLU + Weight-Multiply ONLY
- CUDA kernel: `output[h][j] = max(0, scores[h][j]) * w[h]`
- Eliminates 2-3 elementwise launches per batch (relu + mul)
- Leave .sum(dim=0) as separate torch call (preserve reduction order)
- At B=31: saves ~60-90 launches, ~300-450us

### Step 4: Stream Pipelining
- 2 CUDA streams: overlap batch[i] GEMM+post with batch[i+1] page-gather
- Dequant = memory-bound, GEMM = compute-bound → different SM resources
- Zero correctness risk — batch elements are independent

## DO NOT IMPLEMENT
- Batched GEMM (torch.bmm) — INCORRECT_NUMERICAL proven
- Custom matmul — can't match cuBLAS accumulation
- Custom TopK — can't match CUB radix sort
- Fused sum(dim=0) — different reduction tree breaks topk ordering
- Page pruning — breaks exact topk match

## Test Protocol
After each step: `--quick` then full benchmark. Report per-regime breakdown.

# Session notes — 2026-04-19 through 2026-04-21

Shipping state after three work days on branch `against-2ba145`. Attention is at its local optimum for this algorithmic layout; indexer is at its proven Phase 2c baseline with one further wedge ruled out this session.

## Shipping state (commit `d2233f6`)

| Kernel | Correctness | Approach | Notes |
|---|---|---|---|
| **Indexer** (`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`) | 128/128 PASSED | Phase 2c scoring + vectorized Python topk/gather/remap | Grid is already `(max_num_pages, B)`; no grid-underfill headroom in the bench set. |
| **Attention** (`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`) | 23/23 PASSED | Phase 5b.3 + A1 + Step 2a (2-stage K TMA pipeline) | 4-warp cooperative + mma.sync.m16n8k16 tensor cores for both QK and AV. |

Final bench numbers populated separately from pre-submission verification.

## Recent commit history

```
d2233f6 paired bench: indexer support + short-cohort geomean
f9e3ffa paired bench: A/B infrastructure for instance-variance cancellation
61b3fff ncu_profile: pass KERNEL through function arg, not os.environ
019137e Step 2a: 2-stage K TMA pipeline, thread-parallel double-buffer
0622e16 Phase 2d: explored, algorithm flawed, reverted (128/128 PASS maintained)
89a19e8 A1: tensor-core AV via mma.sync.m16n8k16
```

## What's been tried this session and ruled out

Each with evidence; none should be retried without addressing the root cause listed.

### Attention — 6 ATTEMPTS, 1 COMMITTED

| Step | What | Measurement | Outcome | Root cause if failed |
|---|---|---|---|---|
| **2a** | 2-stage K TMA double-buffer (4-warp math, thread-parallel B′ option) | 3-run mean 4.335× vs 3.924× baseline | **Committed `019137e`** | Infrastructure-valuable; overlap is session-dependent |
| 2b | 3-stage K TMA pipeline | SMEM math: 272 KiB > 228 KiB B200 ceiling | Never implemented | Infeasible at BLOCK_KV=64; would need BLOCK_KV=32 (2× chunks) or O_acc→registers restructure |
| 2e | Eliminate `attn_bf16` SMEM buffer, inline FP32→bf16 cast at AV A-frag load | 3-run mean 4.393× (+1.3% vs 2a) | Reverted | Savings (2 KiB SMEM, 32 syncs/block) within noise floor; L1/TEX bottleneck unaffected |
| 2f | ldmatrix.x4 for QK A + B (batched 2 N-tiles) | Paired geomean 0.9957 (−0.43%), **236 regs/thread** | Reverted | Register pressure from holding 4 u32 B-frag live across 2 mma. Uniform per-workload regression. |
| 2f-v2 | ldmatrix.x2 for QK B (one per N-tile) | Paired geomean 0.9999 (−0.01%), **128 regs/thread** | Reverted | Definitive: ldmatrix provides **no win** on this layout even at neutral register pressure. Compiler was already merging scalar loads. |
| A1 | mma.sync.m16n8k16 BF16 tensor cores for AV (in addition to QK) | +1.82× over 5b.2 in v-old measurement | Committed pre-session as `89a19e8` | — |

**Attention local optimum diagnosis (NCU, long-seq shape T=256 @ 794 µs):**
- L1/TEX throughput: 76.67% — **SMEM-read-bound**
- Compute (SM): 10.35% — tensor cores underused due to stalls on SMEM
- DRAM: 1.96% — fully cached, TMA serves it fine
- Achieved occupancy: 6.25% (limited by 200 KiB SMEM/block → 1 block/SM)
- Waves per SM: 1.73 (partial-wave tail effect)

**Implication:** the mma operand-load SMEM reads are the bottleneck. `ldmatrix` doesn't help (same bytes, compiler already coalesces). The only wedges left are:
1. **A4 — `tcgen05.mma` / UMMA** (deferred; TMEM-resident B bypasses SMEM reads entirely). Changes algorithmic structure, session-length commitment.
2. **Full register-resident P with warp-by-slots AV restructure** (considered in 2e planning; requires chunked N-dim passes with cross-warp O reduction — high complexity).

### Indexer — 2 DIAGNOSIS, 0 COMMITTED THIS SESSION

| Step | What | Measurement | Outcome | Root cause |
|---|---|---|---|---|
| I3 (as specified) | Multi-CTA per batch row for small-B long-seq shapes | — | Never implemented | Current grid is already `(max_num_pages, B)` = multi-CTA per row. The "grid-underfill for small-B long-seq" scenario doesn't exist. |
| Fused topk | Bitonic sort + block_table gather + remap in single kernel launch (replaces Python pipeline's 5-9 launches) | Paired geomean 0.2929 (−70.71%); short-cohort geomean 0.2549 (−74.51%); 128/128 correct | Reverted | Fixed 8192-element bitonic sort is O(log²N) ~= 35 µs per row regardless of actual data size. torch.topk uses CUB radix-select (O(actual N), <2 µs for short rows). Replacement cost > launch-overhead savings. |

**Indexer bottleneck diagnosis:**
- Min speedup (w02: B=1, max_num_pages=3): 7.8×. **NOT** from grid underfill.
- Cause: fixed Python-side post-processing overhead (~25-45 µs of CUDA launch latency for 5-9 `torch.topk`/gather/where/cat calls) dominates microsecond-scale short-seq scoring.
- **To attack this**: the attack must preserve torch.topk's CUB-based selection algorithm efficiency (no hand-rolled bitonic). Candidates not yet explored:
  - **CUDA graph capture** of the whole post-processing pipeline — reduces per-call launch count to 1 replay. Contest rules dependency.
  - **Custom CUDA topk using radix-select** — complex, but O(N) compute matches CUB. Not attempted; Phase 3 experiment with handrolled CUB-style wrappers was parity in prior work.

## What remains as realistic candidates (for future sessions)

**Ordered by expected yield × risk:**

### High value, high risk — session-length commitments

1. **A4: Attention UMMA (`tcgen05.mma`) with TMEM-resident B operand.** Changes the algorithmic structure to avoid SMEM-read bottleneck. NCU confirms this is attention's only remaining wedge. Estimated full session + verification. Moderate chance of 2-3× attention speedup if it lands cleanly.

2. **Full register-resident P fused QK→AV with warp-by-slots AV restructure.** Attempted at design level in Step 2e pre-implementation; path requires chunked N-dim passes + cross-warp O reduction. Could get to 2 blocks/SM on attention. Requires rewriting AV distribution.

### Low value, low risk — fillers if time permits

3. **Indexer BLOCK_KV=32 tile sweep.** Halves per-page KV slot count. Unclear whether it helps given the current grid is already multi-CTA per batch row; may regress via smaller tiles.

4. **Attention `__launch_bounds__` tuning.** Currently `__launch_bounds__(128, 1)`. Might squeeze slight occupancy without any logic change. Quick experiment, low yield.

### Explicitly ruled out (do NOT retry without addressing root cause)

- **Attention ldmatrix (both x4 and x2 tried)** — the compiler already coalesces scalar loads; ldmatrix provides no bandwidth reduction. Ruled out at paired-bench noise floor.
- **Attention attn_bf16 elimination** — SMEM savings too small vs bottleneck; within measurement noise.
- **Attention 3-stage pipeline** — SMEM math doesn't fit at BLOCK_KV=64; would require either BLOCK_KV=32 (2× iterations) or O_acc→registers (larger restructure).
- **Indexer fused topk via bitonic sort** — the replacement algorithm's O(log²N) cost exceeds torch.topk's CUB O(N) cost. A hand-rolled radix-select topk could be retried, but prior Phase 3 experiments already established parity with torch.topk.
- **Indexer Phase 2d streaming-heap topk** — algorithmic correctness bug (bounded-merge fails when pending values exceed frozen threshold). Documented in `kernel_2d.cu`.

## Key gotchas learned this session

### 1. Modal B200 instance heterogeneity is ±25% uniform per session

Arithmetic bench means in the SAME session are highly correlated (same instance), but across sessions they vary ±25% with a uniform multiplier. 3-run aggregate means conflate instance drift with real deltas. **Paired A/B benching in the same session is mandatory** for measuring wins smaller than ~15%.

Harness in `scripts/run_modal_paired_bench.py` — packs baseline and candidate as two solutions in the TraceSet so flashinfer-bench's workload-major execution runs them back-to-back on the same GPU.

### 2. Indexer per-workload noise is ~±30%; attention is ~±1%

Indexer workloads run in microseconds where CUDA event timer resolution is limiting. Individual-workload comparisons are noise-dominated; only aggregate (geomean) or cohort-level comparisons are trustworthy. Short-cohort geomean gate (n=11) introduced in `d2233f6`.

### 3. The `KERNEL` env var does NOT propagate into Modal containers

Pass via function parameter, not `os.environ`. Caught twice: `scripts/ncu_profile.py` (fixed `61b3fff`) and `scripts/run_modal_paired_bench.py` (fixed in `d2233f6`).

### 4. Compiler already coalesces scalar SMEM loads for mma fragments

`ldmatrix` provides instruction-count reduction, not bandwidth reduction, when the compiler can already schedule scalar loads to hit peak SMEM throughput. Test this empirically before committing to an ldmatrix rewrite — the Step 2f experiment showed neutral outcome at matched register pressure. Check NCU `l1tex__throughput` in baseline: if already ≥75%, ldmatrix is unlikely to help.

### 5. Fused single-kernel replacements must account for O(log²N) vs library O(N) algorithm cost

The fused-topk attempt used a bitonic sort (O(log²N)) to replace torch.topk's CUB radix-select (O(actual N)). Launch-overhead savings were real (25-45 µs) but the replacement's compute was larger (~35 µs at padded N=8192). **A fused replacement is only a win when the replacement's algorithm matches or beats the library's asymptotic cost, not just reduces launch count.**

## Infrastructure committed this session (usable in future)

- **`scripts/run_modal_paired_bench.py`** — A/B harness with per-workload ratios, geomean, min/max, cohort filters. Supports both kernels via `KERNEL=` env var.
- **`scripts/probe_ptxas.py`** — direct nvcc compile with `-ptxas-options=-v` to capture register count / SMEM / spill info without going through torch JIT.
- **`scripts/ncu_profile.py`** — Pass-1 NCU triage with per-kernel section sets. Fixed kernel-kind propagation.
- **`indexer/solution/python_baseline_phase2c/`** and **`attention/solution/python_baseline_2a/`** — frozen baselines for paired-bench comparison against future candidates.

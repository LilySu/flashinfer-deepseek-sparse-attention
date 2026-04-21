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

## Architecture diagrams

These three diagrams document the attention kernel's implementation strategy —
data flow through the memory hierarchy, temporal overlap of producer/consumer,
and spatial distribution of tensor-core work across warps.

### Diagram 1 — Attention systems view: memory hierarchy + data flow + techniques

Each arrow is labeled with the technique used (TMA, mma.sync tensor cores,
scalar loads, etc.). Buffer sizes sum to the 200.2 KiB dynamic-SMEM budget.

```
┌──────────────────────── GLOBAL MEMORY (DRAM) ────────────────────────────────┐
│  q_nope [T,16,512] bf16    q_pe [T,16,64] bf16                               │
│  ckv_cache [P*64,512] bf16  kpe_cache [P*64,64] bf16                         │
│  sparse_indices [T,2048] i32                                                 │
└────────┬───────────────────────────┬────────────────────────────────────────┘
         │ ld.global (normal)         │ cp.async.bulk TMA (Blackwell SM100a)
         │   Q staging at              │   73,728 B per chunk = 64 slots ×
         │   kernel entry              │   (512 ckv + 64 kpe) bf16
         │                             │   mbarrier + arrive.expect_tx
         │                             │   2-stage double-buffer
         ▼                             ▼
┌──────────────────────── SHARED MEMORY (200.2 KiB / block, 1 block/SM) ──────┐
│                                                                              │
│  q_concat[16][576]  bf16   ─── 18 KiB ─── staged once per token              │
│                                                                              │
│  k_concat[2][64][576] bf16 ─── 144 KiB ── 2 stages, TMA chunk c+1 overlaps   │
│           │                                compute for chunk c (Diagram 2)   │
│           └─ stage = chunk & 1,  phase = (chunk >> 1) & 1                    │
│                                                                              │
│  logits[16][64]    f32     ─── 4 KiB  ─── QK D-frag spill + softmax staging  │
│  attn_bf16[16][64] bf16    ─── 2 KiB  ─── P after softmax, AV A-frag source  │
│  O_acc[16][512]    f32     ─── 32 KiB ─── online-softmax running output      │
│  m_state, l_state          ─── 128 B  ─── online-softmax per-head state      │
│  k_bar[2]                  ─── 16 B   ─── mbarriers, one per stage           │
│                                                                              │
└────────┬─────────────────────────────────────────────────────────────────────┘
         │ scalar u32 SMEM loads (compiler coalesced; ldmatrix Step 2f/2f-v2
         │ rejected — see Gotcha 4)
         ▼
┌──────────────────────── REGISTERS (128 regs/thread, 0 spills) ──────────────┐
│                                                                              │
│   A frag (4 u32)   ─┐                                                        │
│                     ├─► mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  │
│   B frag (2 u32)   ─┤   ┌────── BF16 Tensor Cores (SM100a) ──────┐           │
│                     │   │  QK: 72 mma/warp/chunk × 4 warps × 32  │           │
│                     ▼   │  AV: 64 mma/warp/chunk × 4 warps × 32  │           │
│   D frag (4 f32)        │      (~17K mma calls/token total)      │           │
│                         └────────────────────────────────────────┘           │
│                                                                              │
└────────┬─────────────────────────────────────────────────────────────────────┘
         │ D written back to SMEM: logits[] (post-QK) or O_acc[] (post-AV)
         │ Online softmax: m = max(m, chunk_max); rescale O by exp(m_old−m_new)
         │ Cast FP32 attn → BF16 for AV mma input
         ▼
┌──────────────────────── GLOBAL MEMORY — OUTPUT ──────────────────────────────┐
│  output[T,16,512] bf16   lse[T,16] f32   ─── final normalize + cast          │
└──────────────────────────────────────────────────────────────────────────────┘

Techniques at a glance:
  • TMA (cp.async.bulk)   — chunks 2 & 3 of each token's K loaded asynchronously
  • mbarrier + expect_tx  — producer/consumer sync with hardware-managed parity
  • BF16 mma.sync tensor cores — both QK and AV (A1 landing gave AV tensor cores)
  • Online softmax (FA-style) — m, l, O accumulated across 32 chunks per token
  • Register-resident accumulators — O stays in regs across chunks (via 64 regs/lane)
```

### Diagram 2 — 2-stage TMA pipeline timing (Step 2a, commit `019137e`)

Producer (tid==0 issues expect_tx; all 128 threads issue cp.async.bulk) runs
one chunk ahead of the consumer so TMA latency is fully hidden behind mma compute.

```
time ────────────────────────────────────────────────────────────────────────►

Producer (TMA unit):
    ┌─TMA─┐   ┌─TMA─┐              ┌─TMA─┐              ┌─TMA─┐
    │ c=0 │   │ c=1 │              │ c=2 │   ...        │c=31 │
    └──┬──┘   └──┬──┘              └──┬──┘              └──┬──┘
     fills s=0 fills s=1            fills s=0            fills s=(31&1)
     → full[0]  → full[1]           → full[0] phase 1    → full[.] phase .
                                      (phase bit flipped)

Consumer (mma + softmax warps):
                    ┌──────────────┐┌──────────────┐┌──────────────┐
                    │ wait full[0] ││ wait full[1] ││ wait full[0] │
                    │   phase=0    ││   phase=0    ││   phase=1    │
                    │              ││              ││              │
                    │ 72 QK mma    ││  ... c=1 ... ││ 72 QK mma    │
                    │ + softmax    ││              ││ + softmax    │
                    │ + 64 AV mma  ││              ││ + 64 AV mma  │
                    │ on k_concat  ││              ││ on k_concat  │
                    │ [stage 0]    ││              ││ [stage 0]    │
                    │              ││              ││              │
                    │ re-issue TMA ││ re-issue TMA ││ re-issue TMA │
                    │ for c+2 →    ││ for c+2 →    ││ for c+2 →    │
                    │ stage 0      ││ stage 1      ││ stage 0      │
                    └──────────────┘└──────────────┘└──────────────┘
                      chunk 0         chunk 1         chunk 2

   ─────────────────►
   prologue: pre-issue TMAs for c=0 → s=0 and c=1 → s=1 so the consumer
              never blocks waiting on the first chunk.

Phase-bit rule for N=2 stages (kernel.cu pipe_phase):
    iter   stage   phase     waits_on
      0      0       0       full[0] phase=0    ← bar[0] init phase is 0
      1      1       0       full[1] phase=0    ← bar[1] init phase is 0
      2      0       1       full[0] phase=1    ← bar[0] flipped after iter 0
      3      1       1       full[1] phase=1
      4      0       0       full[0] phase=0    ← flipped again
     ...

Total TMA issuances per token: 2 (prologue) + (32-2) = 32 = kNumChunks.
  Debug-build asserts this count at kernel exit.
```

### Diagram 3 — Per-warp mma tile layout (how tensor-core work is split)

Four warps share the 128-thread CTA; each owns a disjoint slice of the output
so no cross-warp reduction is needed. QK partitions by N-dim (slots); AV
partitions by N-dim (output channels).

```
QK per chunk: A=Q[M=16 heads, K=576 dims] × B=K[N=64 slots, K=576]^T → D=logits[16,64]

                  ┌──────────────────────────────────────────────────────────┐
   slot N-axis:   │                       64 slots                           │
                  ├─────────────┬─────────────┬─────────────┬────────────────┤
   warp owner:    │   warp 0    │   warp 1    │   warp 2    │   warp 3       │
                  │ slots 0..15 │ slots 16..31│ slots 32..47│ slots 48..63   │
                  │ (2 N-tiles  │ (2 N-tiles  │ (2 N-tiles  │ (2 N-tiles     │
                  │  of 8 slots)│  of 8 slots)│  of 8 slots)│  of 8 slots)   │
                  └─────────────┴─────────────┴─────────────┴────────────────┘

   K-axis sweep:  each warp loops over all 36 K-tiles (each 16 dims wide)
                  per warp per chunk = 2 N-tiles × 36 K-tiles = 72 mma.sync

   Per-lane fragment in ONE mma.m16n8k16.row.col:
     A [16×16 row-major bf16]       B [16×8 col-major bf16]     D [16×8 f32]
     ┌──4 u32 per lane──────┐       ┌─2 u32 per lane─┐         ┌─4 f32──┐
     │ heads t/4, t/4+8     │       │ cols t/4       │         │ t/4,   │
     │ dims  (t%4)*2+       │       │ rows 2*(t%4)+  │         │ t/4+8, │
     │       {0,1,8,9}      │       │      {0,1,8,9} │         │ (t%4)*2│
     └──────────────────────┘       └────────────────┘         └────────┘


AV per chunk: A=P[M=16 heads, K=64 slots] × B=V[K=64 slots, N=512 dims] → O += [16,512]

                  ┌──────────────────────────────────────────────────────────┐
   dim N-axis:    │                    512 output dims                       │
                  ├─────────────┬─────────────┬─────────────┬────────────────┤
   warp owner:    │   warp 0    │   warp 1    │   warp 2    │   warp 3       │
                  │ dims 0..127 │ dims 128..255│dims 256..383│ dims 384..511 │
                  │ (16 N-tiles │ (16 N-tiles │ (16 N-tiles │ (16 N-tiles    │
                  │  of 8 dims) │  of 8 dims) │  of 8 dims) │  of 8 dims)    │
                  └─────────────┴─────────────┴─────────────┴────────────────┘

   K-axis sweep:  each warp loops over all 4 K-tiles (each 16 slots wide)
                  per warp per chunk = 16 N-tiles × 4 K-tiles = 64 mma.sync

   Register-resident per-lane state:
     • 16 N-tiles × 4 f32 = 64 regs of AV accumulator (reused across chunks
       via online-softmax rescale: O *= exp(m_old − m_new) before adding new AV)
     • No cross-warp reduction needed — each warp owns disjoint output dim range

Summary of warp specialization strategy:
  • Compute is data-parallel across warps (no specialized TMA warp;
    thread-parallel double-buffer is "Option B′" from Step 2a planning).
  • Spatially, each warp owns a disjoint output slice — eliminates
    cross-warp reductions that would dominate runtime with SMEM writes.
  • Temporally, the 2-stage TMA pipeline (Diagram 2) overlaps K load with
    compute so memory latency hides behind tensor-core execution.
```

## Infrastructure committed this session (usable in future)

- **`scripts/run_modal_paired_bench.py`** — A/B harness with per-workload ratios, geomean, min/max, cohort filters. Supports both kernels via `KERNEL=` env var.
- **`scripts/probe_ptxas.py`** — direct nvcc compile with `-ptxas-options=-v` to capture register count / SMEM / spill info without going through torch JIT.
- **`scripts/ncu_profile.py`** — Pass-1 NCU triage with per-kernel section sets. Fixed kernel-kind propagation.
- **`indexer/solution/python_baseline_phase2c/`** and **`attention/solution/python_baseline_2a/`** — frozen baselines for paired-bench comparison against future candidates.

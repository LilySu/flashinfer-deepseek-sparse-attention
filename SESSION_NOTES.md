# Session notes — 2026-04-19 through 2026-04-24

Shipping state across four work days on branch `against-2ba145`. Final submission for the 2026-04-24 contest deadline is **`submission-v12`** at commit `ba8e05b`. Attention is at its local optimum for this algorithmic layout; indexer is at its proven Phase 2c baseline. A CuTe DSL UMMA bring-up landed end-to-end correctness on day 4 (**128/128 PASSED at parity speedup vs Phase 2c**, paired-bench geomean 1.0012) but did not clear the +15% paired-bench gate for switching the default. The cute_dsl path is preserved as opt-in on branch `cute-dsl-port` for future work — see Part J below.

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

## Part J — CuTe DSL bring-up (2026-04-24): Phase 1 LANDED, gate not met

Branch `cute-dsl-port` (forked from `against-2ba145`, last commit `adedf0e`). Untouched by submission-v12.

### Phase 0 (toolchain smoke) — PASSED

On eval-matched container `flashinfer/flashinfer-ci-cu132:20260401-2c675fb`:
- All imports clean: `cutlass.cute`, `cutlass.pipeline`, `cutlass.utils`, `cutlass.cute.nvgpu.{cpasync,tcgen05}`, `cutlass.cute.runtime`.
- Trivial `@cute.jit` vector-add: cold JIT 221ms, warm 7us, max_err 0.0.
- MMA atoms present: `MmaF16BF16Op`, `MmaFP8Op` (NOT `MmaF8F6F4Op` as SURVEY assumed), `MmaTF32Op`, `MmaI8Op`, plus block-scaled `MmaMXF4Op` / `MmaMXF8Op` / `MmaMXF4NVF4Op` / `BlockScaledMmaOp`.
- All Pipeline classes + `TmemAllocator` present.

### Phase 1 (indexer FP8 UMMA kernel) — CORRECTNESS LANDED, GATE NOT MET

Near-verbatim port of `fp16_gemm_0.py` with FP8 swap, atom shape `(M=128, N=64, K=32)`, plus heavy Python pre/post-processing wrapper. **128/128 PASSED on real contest workloads.**

**10-iteration bring-up trail** (see commit `adedf0e` for details):
1. `MmaFP8Op` 8-arg signature → 7 args
2. `storage.<int>.ptr` → drop `.ptr` (newer API returns `_Pointer` directly)
3. `TmemAllocator` requires `barrier_for_retrieve=NamedBarrier`
4. Full near-verbatim port of `fp16_gemm_0.py`
5. Grid > 3D (`cute.ceil_div` padding) → hardcoded 3D `(1, 1, L)`
6. **L=1 single-CTA correctness ACHIEVED** (max_err=0.0)
7. `subtile_cnt=4` wrong for our N=64 → set to 1
8. L=8 multi-CTA wrong values
9. L=1 confirmed bit-exact (rule out single-CTA regression)
10. **K-innermost stride fix → L=4 BIT-EXACT all slices** ✓

**The L>1 multi-CTA bug** turned out to be torch tensor stride convention. CuTe wants K-innermost in `mA_mkl` shape `(M, K, L)`, but `torch.randn(M, K, L)` is contiguous with L-innermost (strides `M*K, L, 1`). Fix: allocate as `torch.randn(L, M, K).contiguous()` (strides `M*K, K, 1` — K innermost) and `permute(1, 2, 0)` for the CuTe view — same memory, K-innermost stride pattern. The `fp16_gemm_0.py` reference is 2D so this never came up.

**End-to-end bench results** (CuTe DSL UMMA, INDEXER_BACKEND=cute_dsl):
- `KERNEL=indexer INDEXER_BACKEND=cute_dsl run_modal_bench.py`:
  - **128/128 PASSED**, mean 14.841×, median 13.334×, min 7.040×, max 28.491×
  - vs Phase 2c baseline same session: mean 14.872×, median 13.485×, min 7.264×, max 28.695× — *virtually identical*
- Paired bench (cute_dsl candidate vs Phase 2c baseline, same Modal session):
  - Geomean(new/old): **1.0012 (+0.12%)**
  - Min ratio: 0.9672 / Max ratio: 1.0427
  - Short-cohort geomean: 0.9972 (-0.28%)
  - No workload below 0.95 floor

**Verdict.** Phase 2 commit gate is geomean ≥ 1.15. Achieved +0.12%. **Per the abandonment ladder, ship submission-v12 unchanged.**

**Why the win is hidden.** The CuTe DSL kernel does FP8 UMMA scoring (Blackwell tensor cores engaged), but the Python pre/post-processing wrapper — pre-gather K via block_table, pre-replicate Q across pages, post-process ReLU + Σ_h w[h]·S[h, slot] · scale[slot] — adds enough overhead to nearly cancel the UMMA win. NCU-style breakdown not needed; the math is straightforward.

**Why this is still a major Phase 1 milestone.** The CuTe DSL path is correct end-to-end on all 128 contest workloads, with FP8 tensor cores actually engaging. Unlocks Phase 2 optimization (move pre/post into the kernel) for future sessions.

### API divergences from documentation/example survey (recorded for future sessions)

- `MmaFP8Op` is the FP8 atom name — `MmaF8F6F4Op` does not exist in this version.
- `storage.<int_field>` returns `_Pointer` directly; drop the `.ptr` accessor that the older `fp16_gemm_0.py` reference uses.
- Grid must be ≤ 3D; `cute.ceil_div((*c.shape, 1), tiler)` emits 4D. Use a hardcoded 3D `grid_shape`.
- `subtile_cnt` for `tmem_thr_copy` must satisfy `threads × per-thread-load == tile_size`. For our N=64 with `Ld32x32bOp(Repetition.x64)` (64 fp32/thread), `subtile_cnt=1` (not 4 as in reference for N=256).
- `from_dlpack` must be called **outside** `@cute.jit`. Inside fails with `_Tensor has no __dlpack__`.
- **Critical**: torch's default contiguous layout for 3D tensors `(M, K, L)` has L-innermost. CuTe expects K-innermost. Fix via `torch.randn(L, M, K).contiguous().permute(1, 2, 0)`.

### Artifacts on `cute-dsl-port` (commit `adedf0e`)

- `indexer/solution/python/cute_dsl/layouts.py` — hierarchical layout scaffold.
- `indexer/solution/python/cute_dsl/kernel.py` — `@cute.jit` + `@cute.kernel` FP8 UMMA, multi-CTA batched, bit-exact.
- `indexer/solution/python/cute_dsl/__init__.py` — package marker.
- `indexer/solution/python/binding.py` — adds `_cute_dsl_kernel()` + `INDEXER_BACKEND=cute_dsl` env-var dispatch. **Default stays `phase2c` so submission-v12 behavior is preserved.**
- `scripts/cute_dsl_smoke.py` — toolchain probe.
- `scripts/cute_dsl_indexer_test.py` — single-shape correctness harness.
- `scripts/run_modal_bench.py` and `run_modal_paired_bench.py` — `pip install nvidia-cutlass-dsl`.

### Phase 2 entry point (next session)

The Phase 1 path is correct but at-parity. To unlock the UMMA win:

1. **Move pre-gather K into the kernel.** Currently `binding.py` does `k_fp8[bt_safe]` (torch indexing). In CuTe DSL, this becomes a sparse-gather load — either (a) one TMA-per-page indexed by block_table loaded via cp.async, or (b) `ComposedLayout` + `CustomStride(gather_functor, ...)` per FlashMLA's pattern. Eliminates a B×max_pages×64×128 gathered-K materialization.
2. **Move pre-replicate Q into the kernel.** Currently `q_index_fp8.unsqueeze(1).expand(...)`. In-kernel, each (batch, page) CTA simply loads `Q[batch]` via TMA — no replication needed.
3. **Move post-process (ReLU + Σ_h w[h]·S[h, slot] · scale[slot]) into the epilogue.** Current implementation does this via `torch.relu` + `torch.einsum` + multiply. In-kernel, after TMEM→RMEM copy of S, apply the fused epilogue per the DeepGEMM `sm100_fp8_mqa_logits.cuh` reference (lines 341–374, register-resident `__ffma2_rn` over head pairs).
4. **Grid restructure**: instead of `(1, 1, B*max_pages)` and Python pre-gather, use `(max_pages, B, 1)` with an in-kernel block_table indirection. Matches Phase 2c's grid pattern.

After (1)+(2)+(3) the Python wrapper drops to: input passthrough + topk + remap (Stage B unchanged from Phase 3). Expected speedup: 2-4× over Phase 2c (UMMA engagement, no Python overhead).

## Final submission

**`submission-v12`** at commit `ba8e05b` on `against-2ba145`. 23/23 attention PASSED, 128/128 indexer PASSED, replicated today on Modal B200. Composite arithmetic mean of the two kernel speedups: 9.5–12.8× across Modal B200 instance variance (today's ~9.5×; favorable instance ~12.8×). Top performer per the prior results email is 11.3×; we sit in the same band depending on which Modal instance the eval container lands on.

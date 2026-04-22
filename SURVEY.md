# Library and CUTLASS survey

Reconnaissance document. Not an implementation plan. Consult this before starting any further optimization sessions on the attention or indexer kernels.

## Ruling-derived allowed/disallowed table

Organizer ruling (folded into this survey) translates to: **CUTLASS/CuTe header-level primitives are allowed as building blocks; pre-built kernel instantiations are not**. Litmus test: **who wrote the `__global__`?** If you did and composed CUTLASS primitives inside, it's a building block. If CUTLASS generated the kernel, it's a pre-built kernel.

| Category | Allowed (building blocks) | Disallowed (pre-built kernels / runtime libs) |
|---|---|---|
| Synchronization | `cutlass::arch::NamedBarrier`, `ClusterTransactionBarrier`, `fence_barrier_init`, `umma_arrive` | — |
| Mbarrier / PTX wrappers | `cutlass::arch::*` (barrier, memory) | — |
| CuTe memory primitives | `cute::TMA::*`, `cute::make_tma_copy`, `cute::TmaDescriptor`, `tma::copy` | — |
| CuTe tensor-core primitives | `cute::UMMA::make_instr_desc`, `cute::SM100_MMA_F8F6F4_SS::fma`, `cute::TMEM::Allocator1Sm` | — |
| Layout algebra | `cute::Layout`, `cute::Tensor`, swizzle functors | — |
| Pipeline state machines | `cutlass::PipelineTmaAsync`, `PipelineTmaUmmaAsync` | — |
| Mainloop building blocks | `cutlass::gemm::collective::CollectiveMma` composed into YOUR `__global__` | — |
| Epilogue building blocks | Visitor trees you assemble yourself | — |
| Full GEMM | — | `cutlass::gemm::device::Gemm<...>::operator()` |
| Full kernel builders | — | `CollectiveBuilder<...>` producing a full kernel |
| Runtime vendor kernels | — | `cuBLAS*`, `cuDNN*`, flashinfer runtime, deep_gemm runtime |
| CUB (CUDA toolkit) | **Allowed per my read** — ships with nvcc, used as a primitive inside your own kernel. NOT named in the ruling. | Flag for organizer confirmation if adopting heavily. |

## Contest rule verification

From `memory/project_contest_logistics.md` (2026-04-18): *"Runtime API calls to flashinfer / deep_gemm / cutlass / cuBLAS are NOT allowed at runtime. You MAY copy source from those libraries into your own solution and build/JIT it yourself. CUTLASS/CuTe headers used as building blocks (not pre-built kernel calls) are fine."*

Per 2026-04-21 organizer ruling: scope of "building block" tightened to explicitly allow the CUTLASS/CuTe item categories in the table above. Explicitly disallows `device::Gemm<...>::operator()` and `CollectiveBuilder`-produced full kernels.

Current shipping state (`67fe53e`): compliant — `019137e` attention kernel uses only `cutlass::arch::barrier`-equivalent primitives (hand-written inline PTX for mbarrier/cp.async.bulk) and `mma.sync` (not CUTLASS). Indexer at Phase 2c uses scalar FP32 FMA + hand-written TMA PTX. Both are custom `__global__`s.

## Op 1: Indexer scoring

FP8 Q @ K dot products → ReLU → per-head weighted sum → per-token scale → write dense `logits[B, max_seq_len_kv]`.

- **Q1 — runtime library:** None allowed. `cuBLAS` / deep_gemm runtime forbidden.
- **Q2 — CUTLASS header-only template:** CUTLASS has FP8 GEMM templates (e.g., `cutlass::arch::Mma_SM100`, `cute::SM100_MMA_F8F6F4_SS::fma`) usable as building blocks. No single CUTLASS header captures our exact fused epilogue (ReLU + per-head weighted sum + per-token scale), so we'd compose primitives into our own `__global__`. Yes, usable.
- **Q3 — structural reference:** **DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh`** (sibling paged variant `sm100_fp8_paged_mqa_logits.cuh` for DSA paged KV). MIT license, vendorable. The DeepGEMM file IS a custom `__global__` using `cute::UMMA::*`, `cute::TMA::*`, `ClusterTransactionBarrier`, `TMEM::Allocator1Sm` — i.e., exactly the building-block pattern the ruling permits. UMMA shape: `M=128, N=BLOCK_Q × kNumHeads, K=32`. The ReLU + weighted-sum epilogue (lines 341–374 of DeepGEMM) lives register-resident after `SM100_TMEM_LOAD_32dp32b32x/64x`; uses fused `__ffma2_rn` without SMEM staging.
  - **Shape deviation:** DeepGEMM comment cites typical `h=32, d=64`; our indexer is `h=64, d=128`. Template parameters already cover this — no structural change, just template instantiation.
  - **Effort to adapt: MEDIUM.** Vendor ~5 headers (`common/tma_copy.cuh`, `mma/sm100.cuh`, `ptx/tcgen05.cuh`, `ptx/ld_st.cuh`, plus CUTLASS headers already in flashinfer bundle); port the kernel body with our head/dim constants.
- **Q4 — worth pursuing:** **YES, highest-confidence win in the submission.**
  - NCU on current Phase 2c: **compute SoL 13.6%, no tensor cores used** (scalar FP32 FMA). Hardware is idle on compute.
  - DeepGEMM's kernel is the exact algorithmic template running on Blackwell tensor cores via UMMA.
  - Yield estimate: 3-5× on the compute-bound large-B workloads; unlikely to help min-speedup shapes (those are launch-overhead bound, not compute bound), but the mean-geomean improvement could be substantial.

**Recommendation: custom-kernel-with-reference, DeepGEMM as source.** Vendor sm100_fp8_paged_mqa_logits.cuh and its header dependencies into `indexer/solution/python/third_party/deepgemm_vendored/`; rewrite binding.py to dispatch the new kernel; keep Phase 2c as a fallback for shape ranges the DeepGEMM template doesn't cover.

## Op 2: Top-K post-processing

Replace `torch.topk + gather + remap` Python pipeline (5-9 launches, ~25-45 µs fixed overhead) with a single CUDA kernel.

- **Q1 — runtime library:** None of the forbidden libraries. **CUB (`cub::BlockRadixSort`, `cub::WarpRadixSort`)** is available; my read is **allowed** because CUB ships with the CUDA toolkit and functions as a device-side primitive library you use INSIDE your own kernel (analogous to CUTLASS primitives or Thrust). The ruling text names flashinfer/deep_gemm/cutlass-runtime/cuBLAS but not CUB. **Flag for organizer confirmation before adopting heavily.**
- **Q2 — CUTLASS header-only template:** **No.** CUTLASS is GEMM-focused; no top-K primitive exists.
- **Q3 — structural reference:** CUB `cub::BlockRadixSort<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, ValueT>`. Classic O(N) radix-select over key-value pairs. torch.topk's implementation already uses CUB internally — we'd be writing the same algorithm with one CUDA launch instead of 5-9 Python ops.
- **Q4 — worth pursuing:** **MARGINAL. Skip for this contest window unless Op 1 and Op 3 are already committed.**
  - We already killed the bitonic-fused-topk attempt (commit path removed): −70% geomean because `O(log²N) bitonic` on fixed-8192 padding crushed the launch-overhead savings.
  - A CUB-based topk would be `O(actual N)` — could actually win. But the launch-overhead savings are bounded at ~25-45 µs. For short-cohort workloads (B ≤ 2, max_seq ≤ 512) where this matters, the absolute numbers are already in the 2.4-3.4× range against a reference that itself is fast. Even a 2× speedup on launch overhead would move the geomean maybe 5-10%.
  - Meanwhile Op 1 and Op 3 could move by 3-5×. Opportunity cost is unfavorable.

**Recommendation: SKIP unless Op 1 and Op 3 have landed and there is still session time. Then: probe with a CUB `BlockRadixSort` kernel, low-effort (< 1 session), after organizer confirms CUB status.**

## Op 3: Sparse attention

Decode-MLA with M=16 query heads, ckv=512 + kpe=64 = K=576, sparse gather by topk indices, paged KV, online softmax, FP32 O accumulation.

- **Q1 — runtime library:** None allowed.
- **Q2 — CUTLASS header-only template:** CUTLASS has SM100 FMHA collective primitives (`collective::CollectiveMma`, `epilogue::fusion::*`) usable as building blocks inside a custom `__global__`. Yes, usable — but example 77 itself is not.
- **Q3 — structural reference:** Two candidates.
  - **CUTLASS example 77 Blackwell FMHA (`examples/77_blackwell_fmha/`)** — uses `cutlass::gemm::collective::CollectiveBuilder` to produce a full kernel wrapped by `device::MLA`. Per the ruling, that's a disallowed pre-built kernel. The MLA variant exists (`77_blackwell_mla.cu`, ckv=512 + kpe=64 = 576 structure matches our shape), but:
    - **`static_assert(TileShapeH == 128)` hardcodes H=128.** Our kernel runs at H=16 — would waste 7/8 of UMMA throughput unless we pad+mask or rewrite tile shapes (cascades through CollectiveBuilder).
    - **No sparse-indices path** in MLA example; it's dense paged KV only (uses `gather_tensor.hpp` for page-table indirection, not topk gather).
    - Effort to adapt: **HIGH** + policy-incompatible structure. Not a path.
  - **FlashMLA (`deepseek-ai/FlashMLA`) `csrc/sm100/decode/head64/kernel.cuh`** — raw `__device__` function (custom kernel), exactly our algorithmic shape. Uses `cute::UMMA::*`, `cute::TMEM::Allocator1Sm`, `cutlass::arch::NamedBarrier` — all allowed building blocks per the ruling. V32 model variant has `D_Q = ckv(512) + rope(64) = 576` — **exact shape match**. Sparse gather lives **inside the kernel** via a dedicated producer warp loading topk indices and issuing per-token TMA copies. Warp-specialized: 128 exp + 128 dequant + 32 index producer + 32 KV producer + 32 rope producer.
    - **Shape deviation:** FlashMLA uses `B_H = 64` (heads). Our attention kernel has 16 query heads, and the workload set is decode (1 query position per call). Heads run UMMA M=64 naturally in FlashMLA; for us, H=16 means UMMA at 1/4 utilization unless we stack across tokens. Decode with batch > 1 (B_H = B × 16 ≥ 64 in workloads where B ≥ 4) becomes well-utilized.
    - Effort to adapt: **MEDIUM.** MIT license, ~10 vendored headers, code already tuned for FP8 + sparse + MLA decode. The main implementation work is the H=16 reconciliation (either accept UMMA underutilization or stack heads across tokens into UMMA M=64 groups).
- **Q4 — worth pursuing:** **YES, but lower confidence than Op 1.**
  - NCU: L1/TEX 76.67% (SMEM-read-bound), compute SoL 10.35%. UMMA bypasses SMEM reads for the B operand via TMEM — addresses the exact bottleneck.
  - Yield estimate: 2-4× on large-B workloads (where UMMA utilization is good); little to nothing on B=1 shapes (where UMMA M=64 is 1/4 filled).
  - Risk: higher than Op 1. FlashMLA depends on ~10 headers with their own transitive deps. Warp specialization with 5 distinct roles is complex; correctness surface is large. Session-length commitment.

**Recommendation: custom-kernel-with-reference, FlashMLA as source. Only after Op 1 has landed.** Vendor `csrc/sm100/decode/head64/kernel.cuh` + headers into `attention/solution/python/third_party/flashmla_vendored/`. Keep Step 2a (`019137e`) as the fallback for shapes where UMMA underutilization would regress.

## Ranked recommendations

1. **Op 1 (indexer UMMA, DeepGEMM reference).** Highest-confidence win. Effort: MEDIUM (1-2 sessions including vendoring, bring-up, and correctness). Yield: 3-5× on compute-bound workloads. Path is an exact structural match. **Start here for the next implementation session.**
2. **Op 3 (attention UMMA, FlashMLA reference).** Conditional on Op 1 landing. Effort: MEDIUM-HIGH (2-3 sessions; warp-specialized kernel is large surface). Yield: 2-4× on large-B workloads, neutral-to-risky on B=1 shapes. **Start only after Op 1 is committed and with explicit session-length commitment.**
3. **Op 2 (CUB-based top-K).** Low priority, marginal yield (~5-10% geomean at best). Effort: LOW (< 1 session), contingent on organizer confirming CUB is allowed. **Skip unless Ops 1 and 3 land with remaining session time.**

## Explicitly skipped

- **CUTLASS example 77 adaptation.** Policy-incompatible (`device::MLA` wraps a `CollectiveBuilder` pre-built kernel) and H=128 hardcoded without sparse gather. No salvageable path; read the collective for layout ideas only.
- **Any `cutlass::gemm::device::Gemm<...>` or `CollectiveBuilder`-based approach.** Disallowed by ruling.
- **cuBLAS / cuBLASLt runtime paths.** Disallowed.
- **CUDA graph capture for indexer launch overhead.** Contest harness calls `kernel()` per workload; graph capture across those calls is likely out of our scope and out of contest scope (would require harness changes).
- **Handrolled radix-select topk without CUB.** If CUB is forbidden, this would be the fallback — but the launch-overhead savings alone aren't worth the correctness surface. Better to accept current Python pipeline.
- **Sort-and-coalesce attention pre-pass.** Already tried twice in Phase 5c (both regressed). Documented in `SESSION_NOTES.md`.

## Open question for organizers

Is **CUB** (`cub::BlockRadixSort`, `cub::DeviceSegmentedRadixSort`, generally `#include <cub/cub.cuh>`) considered:
- (a) An allowed primitive library (like CUTLASS/CuTe headers — used INSIDE your own kernel), or
- (b) A runtime library in the "not encouraged" bucket with cuBLAS?

CUB ships with the CUDA toolkit and is maintained by NVIDIA as a device-side primitive library, not as a full-kernel library. My read is (a). Flag if (b) — this eliminates the Op 2 recommendation entirely.

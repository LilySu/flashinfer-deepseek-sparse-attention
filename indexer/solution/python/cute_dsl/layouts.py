"""Hierarchical layout definitions for the DSA indexer (CuTe DSL).

This module is the *search-space backbone* of the CuTe DSL indexer port.
Per the CuTe paper (§3.5 Logical Product / Divide; §2.5.1 Slicing), tiling
and partitioning patterns are expressed as *compositions of layouts at
different levels of a hierarchy*. Every knob we might want to tune — CTA
tile size, UMMA atom shape, stage depth, warp partitioning, SMEM swizzle —
lives at a specific hierarchy level. Tuning = change a constant at that
level; the kernel body does not change.

## Hierarchy levels

| Level | What | Parameter(s)                | Current value        |
|-------|------|-----------------------------|----------------------|
| 0     | Problem (fixed by contest)  | NUM_HEADS, HEAD_DIM, PAGE_SIZE | 64, 128, 64 |
| 1     | CTA tile                    | BLOCK_Q, BLOCK_KV              | 4, 64       |
| 2     | UMMA atom                   | UMMA_M, UMMA_N, UMMA_K         | 128, Q*H, 32|
| 3     | Warp partitioning           | NUM_WARPS, role assignment     | 6: 1T+1M+4E |
| 4     | Pipeline                    | NUM_STAGES                     | 2           |
| 5     | SMEM swizzle                | Derived from atom major-mode   | (auto)      |

Search space for Phase 1 bring-up:
- BLOCK_Q ∈ {1, 2, 4, 8} — 4 is DeepGEMM's default for `sm100_fp8_mqa_logits.cuh`
- NUM_STAGES ∈ {2, 3} — 2-stage works with HEAD_DIM=128 FP8 SMEM budget
- UMMA_M ∈ {64, 128} — M=128 uses 1-CTA atom; M=64 halves UMMA utilization
- NUM_WARPS ∈ {4, 6, 8} — tradeoff between register pressure and producer/
  consumer overlap

Each of these is a single Python constant here. Change the constant, re-run
the paired bench harness, observe the delta.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05

# =============================================================================
# Level 0 — Problem shape (fixed by contest definition)
# =============================================================================

NUM_HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
SCALE_BYTES_PER_SLOT = 4   # per-slot FP32 scale that dequantizes K FP8 -> FP32

# =============================================================================
# Level 1 — CTA tile (search space)
# =============================================================================
# Each CTA processes BLOCK_Q queries × BLOCK_KV slots.  BLOCK_KV is pinned to
# PAGE_SIZE because K is paged (one page-load per CTA preserves TMA contiguity).
# BLOCK_Q is the main knob: larger = better UMMA utilization but more SMEM for Q.

BLOCK_Q = 4
BLOCK_KV = PAGE_SIZE

# =============================================================================
# Level 2 — UMMA atom (search space)
# =============================================================================
# Blackwell tcgen05.mma atom shape. DeepGEMM `sm100_fp8_mqa_logits.cuh` uses
# UMMA_M=128, UMMA_N=BLOCK_Q*NUM_HEADS, UMMA_K=32 (K=32 is FP8's native UMMA K).
# UMMA_M=128 means a single CTA performs a 128-row MMA per issue (no 2-CTA
# paired MMA needed).

UMMA_M = 128
UMMA_N = BLOCK_Q * NUM_HEADS
UMMA_K = 32

MMA_TILE_SHAPE = (UMMA_M, UMMA_N, UMMA_K)

# =============================================================================
# Level 3 — Warp partitioning (search space)
# =============================================================================
# Warp-specialized roles. One TMA producer, one UMMA consumer, several math
# warps for the epilogue (ReLU + per-head-weighted reduce + per-slot scale).
# NUM_WARPS * 32 = threads_per_CTA. Keep a multiple of 4 for TMEM alignment.

NUM_WARPS = 6
THREADS_PER_CTA = NUM_WARPS * 32


class WarpRole:
    """Role assigned per warp index. cute.arch.warp_idx() switches on this."""
    MATH_0 = 0
    MATH_1 = 1
    MATH_2 = 2
    MATH_3 = 3
    MMA_CONSUMER = 4
    TMA_PRODUCER = 5

    @staticmethod
    def from_warp_idx(w):
        # Constant-time role assignment. In CuTe DSL, cute.arch.warp_idx()
        # returns an int that selects the branch via if/elif.
        return w  # identity here; branch on value at kernel site


# =============================================================================
# Level 4 — Pipeline (search space)
# =============================================================================
# Number of stages of the TMA→UMMA pipeline.  Each stage owns one copy of the
# K page in SMEM.  More stages = more TMA-to-UMMA overlap but more SMEM.
#
# SMEM budget (per CTA, in bytes):
#   sQ:      BLOCK_Q × NUM_HEADS × HEAD_DIM × 1 byte  (FP8)
#   sK:      BLOCK_KV × HEAD_DIM × NUM_STAGES × 1 byte (FP8)
#   sScales: BLOCK_KV × NUM_STAGES × 4 bytes (FP32)
#   bars:    NUM_STAGES × 8 bytes (mbarrier handles)
# For BLOCK_Q=4, NUM_HEADS=64, HEAD_DIM=128, BLOCK_KV=64, NUM_STAGES=2:
#   sQ=32 KiB, sK=16 KiB, sScales=0.5 KiB.  Fits easily in B200's 228 KiB.

NUM_STAGES = 2


# =============================================================================
# Level 5 — SMEM layouts (derived)
# =============================================================================
# Layouts are declared as cute.make_layout(shape, stride) at module scope so
# the layout algebra can be traced at @cute.jit compile time.  Swizzle is
# applied at the TMA-atom construction site (make_tiled_tma_atom_A/B), using
# the MMA atom's major-mode tag — CuTe DSL handles this via the atom's
# `make_fragment_A / make_fragment_B` methods which bake in the correct
# swizzle pattern.


def sQ_layout():
    """Q tile in SMEM: (BLOCK_Q, NUM_HEADS, HEAD_DIM), K-contiguous."""
    return cute.make_layout(
        (BLOCK_Q, NUM_HEADS, HEAD_DIM),
        stride=(NUM_HEADS * HEAD_DIM, HEAD_DIM, 1),
    )


def sK_layout():
    """K tile in SMEM: (BLOCK_KV, HEAD_DIM, NUM_STAGES), K-contiguous.

    NUM_STAGES is the outer-most mode; tma_atom issues into stage[s] chunk.
    """
    return cute.make_layout(
        (BLOCK_KV, HEAD_DIM, NUM_STAGES),
        stride=(HEAD_DIM, 1, BLOCK_KV * HEAD_DIM),
    )


def sScales_layout():
    """Per-slot FP32 scales: (BLOCK_KV, NUM_STAGES)."""
    return cute.make_layout(
        (BLOCK_KV, NUM_STAGES),
        stride=(1, BLOCK_KV),
    )


def sWeights_layout():
    """Per-head FP32 weights: (NUM_HEADS,). Loaded once per CTA."""
    return cute.make_layout((NUM_HEADS,), stride=(1,))


# =============================================================================
# MMA atom factory
# =============================================================================


def make_indexer_tiled_mma():
    """Construct the FP8 UMMA atom for the indexer scoring GEMM.

    Exact API name may need adjustment against installed CuTe DSL version.
    The canonical form per `examples/python/CuTeDSL/blackwell/` tutorials:
    """
    op = tcgen05.MmaF8F6F4Op(
        io_dtype=cutlass.Float8_E4M3,
        acc_dtype=cutlass.Float32,
        mma_tile_shape=MMA_TILE_SHAPE,
        cta_group=tcgen05.CtaGroup.ONE,
        a_source=tcgen05.OperandSource.SMEM,
        b_source=tcgen05.OperandSource.SMEM,
        a_major_mode=tcgen05.OperandMajorMode.K,
        b_major_mode=tcgen05.OperandMajorMode.K,
    )
    return cute.make_tiled_mma(op)


# =============================================================================
# Compile-time sanity checks
# =============================================================================
# These are asserts that fire at Python import time — they catch hierarchy
# inconsistencies before we even try to JIT-compile.

assert NUM_HEADS % UMMA_K == 0 or UMMA_K % NUM_HEADS == 0, \
    "UMMA_K must divide or be divided by NUM_HEADS for clean A-frag tiling"

assert BLOCK_KV == PAGE_SIZE, \
    "BLOCK_KV must equal PAGE_SIZE to preserve TMA contiguity per page"

assert UMMA_N == BLOCK_Q * NUM_HEADS, \
    "UMMA_N expects Q×H rows folded into the N dimension (DeepGEMM convention)"

assert NUM_WARPS >= 3, \
    "Need at least 1 TMA + 1 MMA + 1 math warp"

assert THREADS_PER_CTA <= 1024, \
    "CUDA block size limit"


# =============================================================================
# Introspection helpers (for search-space exploration)
# =============================================================================


def smem_budget_bytes() -> int:
    """Total SMEM required per CTA under current hierarchy settings."""
    sQ_bytes = BLOCK_Q * NUM_HEADS * HEAD_DIM * 1          # FP8
    sK_bytes = BLOCK_KV * HEAD_DIM * NUM_STAGES * 1        # FP8
    sScales_bytes = BLOCK_KV * NUM_STAGES * 4              # FP32
    sWeights_bytes = NUM_HEADS * 4                         # FP32
    bar_bytes = NUM_STAGES * 8 + 32                        # mbarriers + TMEM ptr
    tmem_tile_bytes = 0  # TMEM is separate from SMEM on Blackwell
    return sQ_bytes + sK_bytes + sScales_bytes + sWeights_bytes + bar_bytes


def mma_utilization() -> float:
    """Fraction of a single UMMA atom's M-dimension that we actually use."""
    rows_used = BLOCK_Q * NUM_HEADS
    return rows_used / UMMA_M


def search_space_summary():
    """Human-readable printout of the current point in the search space."""
    return {
        "Level 1 (CTA tile)": {"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV},
        "Level 2 (UMMA atom)": {
            "M": UMMA_M, "N": UMMA_N, "K": UMMA_K,
            "utilization": f"{mma_utilization():.0%}",
        },
        "Level 3 (warps)": {"NUM_WARPS": NUM_WARPS, "threads": THREADS_PER_CTA},
        "Level 4 (pipeline)": {"NUM_STAGES": NUM_STAGES},
        "SMEM bytes / CTA": smem_budget_bytes(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(search_space_summary(), indent=2))

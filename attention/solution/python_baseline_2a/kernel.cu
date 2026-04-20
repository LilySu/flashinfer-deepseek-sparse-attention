// DSA sparse attention — Phase 5b.3 + A1 + Step 2a: 2-stage K pipeline.
//
// Builds on 5b.3 + A1 by double-buffering k_concat across 2 stages so
// that TMA for chunk c+1 overlaps compute for chunk c. Option B′:
// thread-parallel double-buffer — all 4 warps continue to do compute on
// the active stage; TMA issuance remains data-parallel across 128
// threads, with a single `arrive.expect_tx` from tid==0 per stage cycle.
//
// Pipeline (kNumStages=2):
//   Prologue: init full[0], full[1]; issue TMA chunk 0→stage 0, chunk 1→stage 1.
//   Loop chunk = 0..N-1:
//     stage = chunk & 1;  phase = (chunk >> 1) & 1
//     wait full[stage] phase;  compute on stage
//     if (chunk + kNumStages < N): re-issue TMA for chunk+2 → stage
// Total TMA issuances per block: 2 + (N-2) = N = kNumChunks. Debug build
// asserts this at kernel exit.
//
// Extends Phase 5b.3 by replacing the scalar FP32 FMA AV computation
// with warp-level mma.sync.m16n8k16 BF16 tensor cores. After softmax
// the FP32 attention probabilities are converted to BF16 in a dedicated
// SMEM buffer (attn_bf16) and combined with the ckv half of k_concat
// via M=16, N=8, K=16 mma.sync shapes.
//
// Per warp: N-tiles = 128 dims / 8 = 16, K-tiles = 64 slots / 16 = 4,
// so 16 × 4 = 64 AV mma.sync calls per warp per chunk.
//
// Per-warp work:
//   N = 64 slots  → split into 4 warps × 16 slots = 2 N-tiles of 8 per warp
//   K = 576 dims  → 36 K-tiles of 16
//   Per warp per chunk: 2 × 36 = 72 mma.sync calls
//
// AV (output update) remains scalar FP32 FMA — tensor-core AV is a
// future phase. Online softmax state (m, l) in FP32 SMEM (unchanged).
//
// Register layout per lane (m16n8k16 row.col f32.bf16.bf16.f32):
//   A (M=16, K=16) row-major, 4 × uint32 per lane:
//     reg[0] = (A[t/4,  t%4*2+0], A[t/4,  t%4*2+1])  — 2 bf16 packed
//     reg[1] = (A[t/4+8,t%4*2+0], A[t/4+8,t%4*2+1])
//     reg[2] = (A[t/4,  t%4*2+8], A[t/4,  t%4*2+9])
//     reg[3] = (A[t/4+8,t%4*2+8], A[t/4+8,t%4*2+9])
//   B (K=16, N=8) col-major, 2 × uint32 per lane:
//     col_of_lane = t / 4,  row_pair_base = 2*(t%4)
//     reg[0] = (B[row_pair_base+0, col], B[row_pair_base+1, col])
//     reg[1] = (B[row_pair_base+8, col], B[row_pair_base+9, col])
//   D (M=16, N=8) f32, 4 × float per lane:
//     val[0..3] at (rows {t/4, t/4+8}, cols {t%4*2, t%4*2+1})

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cassert>
#include <cstdint>
#include <cmath>

namespace {

constexpr int kQoHeads    = 16;
constexpr int kHeadDimCkv = 512;
constexpr int kHeadDimKpe = 64;
constexpr int kQKDim      = kHeadDimCkv + kHeadDimKpe;   // 576
constexpr int kTopK       = 2048;
constexpr int kBlockKv    = 64;
constexpr int kNumChunks  = kTopK / kBlockKv;   // 32
constexpr int kNumStages  = 2;

static_assert(kNumChunks >= kNumStages,
    "Pipeline requires at least kNumStages chunks");

constexpr int kThreads    = 128;
constexpr int kWarps      = 4;
constexpr int kSlotsPerWarp = kBlockKv / kWarps;   // 16
constexpr int kNTilesPerWarp = kSlotsPerWarp / 8;  // 2
constexpr int kKTiles     = kQKDim / 16;            // 36

struct AttnSmem {
    __nv_bfloat16 q_concat[kQoHeads][kQKDim];                 // 18 KiB
    __nv_bfloat16 k_concat[kNumStages][kBlockKv][kQKDim];     // 2 × 72 KiB
    float  logits[kQoHeads][kBlockKv];                         // 4 KiB
    __nv_bfloat16 attn_bf16[kQoHeads][kBlockKv];               // 2 KiB
    float  m_state[kQoHeads];
    float  l_state[kQoHeads];
    float  O_acc[kQoHeads][kHeadDimCkv];                       // 32 KiB
    alignas(8) uint64_t k_bar[kNumStages];
};

// Phase-bit helper for N=2-stage mbarrier pipeline with 0-init parity.
//   stage(iter) = iter & 1
//   phase(iter) = (iter >> 1) & 1
// Worked example (N=2, 10 iters):
//   iter  stage  phase   wait_on
//    0      0      0     full[0] phase=0
//    1      1      0     full[1] phase=0
//    2      0      1     full[0] phase=1
//    3      1      1     full[1] phase=1
//    4      0      0     full[0] phase=0
//    5      1      0     full[1] phase=0
//    6      0      1     full[0] phase=1
//    7      1      1     full[1] phase=1
//    8      0      0     full[0] phase=0
//    9      1      0     full[1] phase=0
// Each stage's bar visits every 2 iters; parity flips on each visit.
__device__ __forceinline__ uint32_t pipe_stage(int iter) { return iter & 1; }
__device__ __forceinline__ uint32_t pipe_phase(int iter) { return (iter >> 1) & 1; }

// AV constants (N=512 dims split across 4 warps = 128 dims/warp = 16 N-tiles).
constexpr int kAVDimsPerWarp  = kHeadDimCkv / kWarps;    // 128
constexpr int kAVNTilesPerWarp = kAVDimsPerWarp / 8;      // 16
constexpr int kAVKTiles       = kBlockKv / 16;             // 4

constexpr uint32_t kTmaBytesPerChunk =
    kBlockKv * (kHeadDimCkv * 2 + kHeadDimKpe * 2);  // 73728

// ---- mbarrier / cp.async.bulk helpers ----
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    uint32_t r;
    asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
                 : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 : : "r"(smem_addr(bar)), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 : : "r"(smem_addr(bar)), "r"(tx));
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t done;
    do {
        asm volatile("{ .reg .pred p;"
                     "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                     "  selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(smem_addr(bar)), "r"(phase));
    } while (!done);
}
__device__ __forceinline__ void cp_async_bulk_g2s(
    void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar
) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        : : "r"(smem_addr(smem_dst)), "l"(gmem_src),
            "r"(bytes), "r"(smem_addr(bar))
    );
}

// ---- mma.sync.m16n8k16 row.col f32.bf16.bf16.f32 ----
__device__ __forceinline__ void mma_m16n8k16_bf16_f32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float& d0, float& d1, float& d2, float& d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        " {%0, %1, %2, %3},"
        " {%4, %5, %6, %7},"
        " {%8, %9},"
        " {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

// Read a bf16 pair (2 consecutive bf16 = uint32_t) from SMEM row-major tensor.
// smem: base bf16 pointer, stride: row stride in BF16 elements,
// row/col: logical position of the first of the two bf16 values.
__device__ __forceinline__ uint32_t
load_bf16_pair(const __nv_bfloat16* smem, int stride, int row, int col) {
    const __nv_bfloat16* p = smem + row * stride + col;
    return *reinterpret_cast<const uint32_t*>(p);
}

// Pack two BF16 values from arbitrary memory locations into a single uint32
// with lo in bits [0..15] and hi in bits [16..31]. Needed for mma.sync B
// col-major when the row-pair spans non-adjacent SMEM bytes (i.e., AV where
// B[k, n] = k_concat[slot=k, dim=n] and the pair crosses slots, stride 576).
__device__ __forceinline__ uint32_t pack_bf16_pair(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    uint16_t lo16 = *reinterpret_cast<uint16_t*>(&lo);
    uint16_t hi16 = *reinterpret_cast<uint16_t*>(&hi);
    return static_cast<uint32_t>(lo16) | (static_cast<uint32_t>(hi16) << 16);
}

__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, off);
    return x;
}

// Issues cp.async.bulk for one chunk into k_concat[stage].
// Confirmation-2: exactly one thread (tid==0) calls arrive_expect_tx per
// invocation. All 128 threads then participate in the cp.async.bulk data
// transfers (each contributing bytes that decrement the barrier's tx
// counter). Caller is responsible for eventually waiting via mbarrier_wait.
__device__ __forceinline__ void issue_k_tma_for_chunk(
    AttnSmem& smem,
    int stage,
    int chunk_off,
    const int32_t* __restrict__ sparse_idx,
    int64_t sp_base,
    const __nv_bfloat16* __restrict__ ckv_flat,
    const __nv_bfloat16* __restrict__ kpe_flat,
    int tid
) {
    if (tid == 0) {
        mbarrier_arrive_expect_tx(&smem.k_bar[stage], kTmaBytesPerChunk);
    }
    __syncthreads();  // expect_tx happens-before any cp.async.bulk

    for (int i = tid; i < 2 * kBlockKv; i += kThreads) {
        const bool is_kp = (i >= kBlockKv);
        const int s = is_kp ? (i - kBlockKv) : i;
        int32_t idx = sparse_idx[sp_base + chunk_off + s];
        int safe_idx = (idx >= 0) ? idx : 0;
        if (is_kp) {
            cp_async_bulk_g2s(
                &smem.k_concat[stage][s][kHeadDimCkv],
                kpe_flat + static_cast<int64_t>(safe_idx) * kHeadDimKpe,
                kHeadDimKpe * sizeof(__nv_bfloat16),
                &smem.k_bar[stage]);
        } else {
            cp_async_bulk_g2s(
                &smem.k_concat[stage][s][0],
                ckv_flat + static_cast<int64_t>(safe_idx) * kHeadDimCkv,
                kHeadDimCkv * sizeof(__nv_bfloat16),
                &smem.k_bar[stage]);
        }
    }
}

__global__ void __launch_bounds__(kThreads, 1)
attention_kernel_phase5b(
    const __nv_bfloat16* __restrict__ q_nope,     // [T, 16, 512]
    const __nv_bfloat16* __restrict__ q_pe,       // [T, 16, 64]
    const __nv_bfloat16* __restrict__ ckv_flat,   // [P*64, 512]
    const __nv_bfloat16* __restrict__ kpe_flat,   // [P*64, 64]
    const int32_t*       __restrict__ sparse_idx, // [T, 2048]
    __nv_bfloat16*       __restrict__ output,     // [T, 16, 512]
    float*               __restrict__ lse,        // [T, 16]
    float sm_scale,
    int num_tokens
) {
    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lane = tid & 31;
    if (t >= num_tokens) return;

    extern __shared__ uint8_t dyn_smem_bytes[];
    AttnSmem& smem = *reinterpret_cast<AttnSmem*>(dyn_smem_bytes);

    // --- Stage Q concatenated: q_concat[h, 0..511] = qn[h], [512..575] = qp[h] ---
    const int64_t qn_base = static_cast<int64_t>(t) * kQoHeads * kHeadDimCkv;
    const int64_t qp_base = static_cast<int64_t>(t) * kQoHeads * kHeadDimKpe;
    for (int i = tid; i < kQoHeads * kHeadDimCkv; i += kThreads) {
        const int h = i / kHeadDimCkv;
        const int d = i % kHeadDimCkv;
        smem.q_concat[h][d] = q_nope[qn_base + i];
    }
    for (int i = tid; i < kQoHeads * kHeadDimKpe; i += kThreads) {
        const int h = i / kHeadDimKpe;
        const int d = i % kHeadDimKpe;
        smem.q_concat[h][kHeadDimCkv + d] = q_pe[qp_base + i];
    }
    for (int i = tid; i < kQoHeads; i += kThreads) {
        smem.m_state[i] = -INFINITY;
        smem.l_state[i] = 0.0f;
    }
    for (int i = tid; i < kQoHeads * kHeadDimCkv; i += kThreads) {
        smem.O_acc[i / kHeadDimCkv][i % kHeadDimCkv] = 0.0f;
    }

    // Init both stage barriers before any TMA issue.
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < kNumStages; ++s) mbarrier_init(&smem.k_bar[s], 1);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

#ifndef NDEBUG
    __shared__ int debug_tma_count;
    if (tid == 0) debug_tma_count = 0;
    __syncthreads();
#endif

    const int64_t sp_base = static_cast<int64_t>(t) * kTopK;

    // Prologue: fill the first kNumStages stages. Each issue is guarded by
    // s < kNumChunks so shapes with fewer chunks than stages are safe.
    #pragma unroll
    for (int s = 0; s < kNumStages; ++s) {
        if (s < kNumChunks) {
            issue_k_tma_for_chunk(smem, s, s * kBlockKv,
                                  sparse_idx, sp_base, ckv_flat, kpe_flat, tid);
#ifndef NDEBUG
            if (tid == 0) debug_tma_count++;
#endif
        }
    }

    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
        const int chunk_off  = chunk * kBlockKv;
        const int stage      = pipe_stage(chunk);
        const uint32_t phase = pipe_phase(chunk);

        // Wait for this stage's K to be fully delivered.
        mbarrier_wait(&smem.k_bar[stage], phase);
        __syncthreads();

        // --- QK via mma.sync tensor cores ----------------------------------
        // Each warp owns 16 slots (2 N-tiles of 8 each). Warp w covers
        // slots [w*16, w*16+16). Accumulators: 2 N-tiles × 4 f32 per lane.
        float acc[kNTilesPerWarp][4];
        #pragma unroll
        for (int n = 0; n < kNTilesPerWarp; ++n) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[n][j] = 0.0f;
        }

        const int slot_base_for_warp = wid * kSlotsPerWarp;

        // Loop K dim in tiles of 16.
        for (int kt = 0; kt < kKTiles; ++kt) {
            const int k_base = kt * 16;

            // ---- Load A fragment (16 heads × 16 K-dims) ----
            // Q is [16 heads, 576 dims] row-major in SMEM (stride = kQKDim).
            const int a_row0 = lane / 4;           // 0..7
            const int a_row1 = a_row0 + 8;         // 8..15
            const int a_colA = (lane & 3) * 2;     // 0, 2, 4, 6
            const int a_colB = a_colA + 8;         // 8, 10, 12, 14
            const uint32_t a0 = load_bf16_pair(&smem.q_concat[0][0], kQKDim,
                                                a_row0, k_base + a_colA);
            const uint32_t a1 = load_bf16_pair(&smem.q_concat[0][0], kQKDim,
                                                a_row1, k_base + a_colA);
            const uint32_t a2 = load_bf16_pair(&smem.q_concat[0][0], kQKDim,
                                                a_row0, k_base + a_colB);
            const uint32_t a3 = load_bf16_pair(&smem.q_concat[0][0], kQKDim,
                                                a_row1, k_base + a_colB);

            // ---- For each N-tile (8 slots), load B fragment + mma -----
            #pragma unroll
            for (int nt = 0; nt < kNTilesPerWarp; ++nt) {
                const int slot_col = slot_base_for_warp + nt * 8 + (lane / 4);
                const int b_row_base = 2 * (lane & 3);
                // B[k_row, n_col] = k_concat[slot_col, k_base + k_row]
                const uint32_t b0 = load_bf16_pair(&smem.k_concat[stage][0][0], kQKDim,
                                                    slot_col, k_base + b_row_base);
                const uint32_t b1 = load_bf16_pair(&smem.k_concat[stage][0][0], kQKDim,
                                                    slot_col, k_base + b_row_base + 8);

                mma_m16n8k16_bf16_f32(a0, a1, a2, a3,
                                      b0, b1,
                                      acc[nt][0], acc[nt][1],
                                      acc[nt][2], acc[nt][3]);
            }
        }

        // --- Write logits from D fragments to SMEM (with sm_scale and -inf mask) ---
        // D layout per lane: (rows {lane/4, lane/4+8}, cols {lane%4*2, lane%4*2+1}).
        // For each N-tile, write 4 FP32 values to smem.logits.
        #pragma unroll
        for (int nt = 0; nt < kNTilesPerWarp; ++nt) {
            const int n_base = slot_base_for_warp + nt * 8;
            const int col0 = n_base + (lane & 3) * 2;
            const int col1 = col0 + 1;
            const int row0 = lane / 4;
            const int row1 = row0 + 8;
            // Check if slots are valid (sparse_idx == -1 → -inf).
            int32_t idx0 = sparse_idx[sp_base + chunk_off + col0];
            int32_t idx1 = sparse_idx[sp_base + chunk_off + col1];
            float v00 = (idx0 >= 0) ? acc[nt][0] * sm_scale : -INFINITY;
            float v01 = (idx1 >= 0) ? acc[nt][1] * sm_scale : -INFINITY;
            float v10 = (idx0 >= 0) ? acc[nt][2] * sm_scale : -INFINITY;
            float v11 = (idx1 >= 0) ? acc[nt][3] * sm_scale : -INFINITY;
            smem.logits[row0][col0] = v00;
            smem.logits[row0][col1] = v01;
            smem.logits[row1][col0] = v10;
            smem.logits[row1][col1] = v11;
        }
        __syncthreads();

        // --- Softmax state update + O rescale ---
        __shared__ float rescale_smem[kQoHeads];
        if (tid < kQoHeads) {
            const int h = tid;
            const float m_old = smem.m_state[h];
            float chunk_max = -INFINITY;
            for (int s = 0; s < kBlockKv; ++s)
                chunk_max = fmaxf(chunk_max, smem.logits[h][s]);
            const float m_new = fmaxf(m_old, chunk_max);
            const float rescale = isfinite(m_old) ? expf(m_old - m_new) : 0.0f;
            rescale_smem[h] = rescale;

            float new_l = smem.l_state[h] * rescale;
            for (int s = 0; s < kBlockKv; ++s) {
                const float v = smem.logits[h][s];
                const float e = isfinite(v) ? expf(v - m_new) : 0.0f;
                smem.logits[h][s] = e;
                new_l += e;
            }
            smem.m_state[h] = m_new;
            smem.l_state[h] = new_l;
        }
        __syncthreads();

        // --- A1: cast FP32 attn probabilities to BF16 for mma.sync AV ---
        for (int i = tid; i < kQoHeads * kBlockKv; i += kThreads) {
            const int h = i / kBlockKv;
            const int s = i % kBlockKv;
            smem.attn_bf16[h][s] = __float2bfloat16(smem.logits[h][s]);
        }
        __syncthreads();

        // --- AV via mma.sync.m16n8k16 BF16 tensor cores -----------------
        // Per warp owns 128 of the 512 output dims: warp w covers dims
        // [w*128, w*128+128). 16 N-tiles of 8 dims × 4 K-tiles of 16 slots.
        const int dim_base_for_warp = wid * kAVDimsPerWarp;

        float av_acc[kAVNTilesPerWarp][4];
        #pragma unroll
        for (int nt = 0; nt < kAVNTilesPerWarp; ++nt) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) av_acc[nt][j] = 0.0f;
        }

        // Loop K (slot) in tiles of 16.
        #pragma unroll
        for (int kt = 0; kt < kAVKTiles; ++kt) {
            const int k_slot_base = kt * 16;

            // ---- Load A fragment from attn_bf16[16 heads, kt_slots*16] ----
            // A row-major; pair across K (slot) axis → contiguous in attn_bf16
            // (row stride = kBlockKv = 64 bf16). Use load_bf16_pair.
            const int a_row0 = lane / 4;
            const int a_row1 = a_row0 + 8;
            const int a_colA = (lane & 3) * 2;
            const int a_colB = a_colA + 8;
            const uint32_t a0 = load_bf16_pair(&smem.attn_bf16[0][0], kBlockKv,
                                                a_row0, k_slot_base + a_colA);
            const uint32_t a1 = load_bf16_pair(&smem.attn_bf16[0][0], kBlockKv,
                                                a_row1, k_slot_base + a_colA);
            const uint32_t a2 = load_bf16_pair(&smem.attn_bf16[0][0], kBlockKv,
                                                a_row0, k_slot_base + a_colB);
            const uint32_t a3 = load_bf16_pair(&smem.attn_bf16[0][0], kBlockKv,
                                                a_row1, k_slot_base + a_colB);

            // ---- For each N-tile (8 output dims), load B + mma ----
            // B [K=16 slots, N=8 dims]. B[k_row, n_col] = k_concat[k_slot_base + k_row, dim_base + n_col].
            // In row-major k_concat, pair-across-rows (K axis) is stride kQKDim → NOT adjacent,
            // so use pack_bf16_pair with 2 separate loads per reg.
            const int row_k_base = 2 * (lane & 3);  // 0, 2, 4, 6
            #pragma unroll
            for (int nt = 0; nt < kAVNTilesPerWarp; ++nt) {
                const int n_base = dim_base_for_warp + nt * 8;
                const int dim_col = n_base + (lane / 4);  // 0..7 within 8-col tile

                const __nv_bfloat16 b00 =
                    smem.k_concat[stage][k_slot_base + row_k_base + 0][dim_col];
                const __nv_bfloat16 b01 =
                    smem.k_concat[stage][k_slot_base + row_k_base + 1][dim_col];
                const __nv_bfloat16 b10 =
                    smem.k_concat[stage][k_slot_base + row_k_base + 8][dim_col];
                const __nv_bfloat16 b11 =
                    smem.k_concat[stage][k_slot_base + row_k_base + 9][dim_col];
                const uint32_t b0 = pack_bf16_pair(b00, b01);
                const uint32_t b1 = pack_bf16_pair(b10, b11);

                mma_m16n8k16_bf16_f32(a0, a1, a2, a3,
                                      b0, b1,
                                      av_acc[nt][0], av_acc[nt][1],
                                      av_acc[nt][2], av_acc[nt][3]);
            }
        }

        // --- Rescale existing O_acc and add AV contribution ---------------
        // D layout per lane (same as QK D): rows {lane/4, lane/4+8} ×
        // cols {lane%4*2, lane%4*2+1}, so each lane writes 4 (h, d) entries
        // per N-tile. Warps own disjoint dim ranges → no cross-warp race;
        // within warp each (h, d) is owned by exactly one lane.
        #pragma unroll
        for (int nt = 0; nt < kAVNTilesPerWarp; ++nt) {
            const int n_base = dim_base_for_warp + nt * 8;
            const int d_col0 = n_base + (lane & 3) * 2;
            const int d_col1 = d_col0 + 1;
            const int d_row0 = lane / 4;
            const int d_row1 = d_row0 + 8;
            const float rs0 = rescale_smem[d_row0];
            const float rs1 = rescale_smem[d_row1];
            smem.O_acc[d_row0][d_col0] =
                smem.O_acc[d_row0][d_col0] * rs0 + av_acc[nt][0];
            smem.O_acc[d_row0][d_col1] =
                smem.O_acc[d_row0][d_col1] * rs0 + av_acc[nt][1];
            smem.O_acc[d_row1][d_col0] =
                smem.O_acc[d_row1][d_col0] * rs1 + av_acc[nt][2];
            smem.O_acc[d_row1][d_col1] =
                smem.O_acc[d_row1][d_col1] * rs1 + av_acc[nt][3];
        }
        __syncthreads();  // all threads finished reading k_concat[stage]

        // Pre-fetch chunk + kNumStages into the stage we just drained.
        // Confirmation-3: explicit guard so we never issue a TMA for an
        // out-of-range chunk (would pass rtol=1e-2 silently for sparse attn).
        if (chunk + kNumStages < kNumChunks) {
            issue_k_tma_for_chunk(smem, stage,
                                  (chunk + kNumStages) * kBlockKv,
                                  sparse_idx, sp_base, ckv_flat, kpe_flat, tid);
#ifndef NDEBUG
            if (tid == 0) debug_tma_count++;
#endif
        }
    }

#ifndef NDEBUG
    __syncthreads();
    if (tid == 0) assert(debug_tma_count == kNumChunks);
#endif

    // --- Final normalize + write ---
    for (int i = tid; i < kQoHeads * kHeadDimCkv; i += kThreads) {
        const int h = i / kHeadDimCkv;
        const int d = i % kHeadDimCkv;
        const float denom = (smem.l_state[h] > 0.0f) ? smem.l_state[h] : 1.0f;
        output[qn_base + i] = __float2bfloat16(smem.O_acc[h][d] / denom);
    }
    if (tid < kQoHeads) {
        const int h = tid;
        const float l = smem.l_state[h];
        if (l > 0.0f) {
            constexpr float kInvLog2 = 1.4426950408889634f;
            lse[static_cast<int64_t>(t) * kQoHeads + h] =
                smem.m_state[h] * kInvLog2 + log2f(l);
        } else {
            lse[static_cast<int64_t>(t) * kQoHeads + h] = -INFINITY;
        }
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// Host entry
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> dsa_sparse_attention(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    double sm_scale
) {
    TORCH_CHECK(q_nope.is_cuda());
    TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(ckv_cache.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(kpe_cache.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(sparse_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(q_nope.size(1) == kQoHeads);
    TORCH_CHECK(q_nope.size(2) == kHeadDimCkv);
    TORCH_CHECK(q_pe.size(2) == kHeadDimKpe);
    TORCH_CHECK(sparse_indices.size(1) == kTopK);

    const int64_t num_tokens = q_nope.size(0);
    auto out_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(q_nope.device());
    auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device());
    auto output = torch::empty({num_tokens, kQoHeads, kHeadDimCkv}, out_opts);
    auto lse = torch::full({num_tokens, kQoHeads}, -INFINITY, lse_opts);
    if (num_tokens == 0) return {output, lse};

    auto ckv_flat = ckv_cache.reshape({-1, kHeadDimCkv});
    auto kpe_flat = kpe_cache.reshape({-1, kHeadDimKpe});

    const int smem_bytes = sizeof(AttnSmem);
    static bool smem_opt_set = false;
    if (!smem_opt_set) {
        cudaFuncSetAttribute(
            attention_kernel_phase5b,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        smem_opt_set = true;
    }

    const dim3 grid(static_cast<unsigned>(num_tokens));
    const dim3 block(kThreads);

    attention_kernel_phase5b<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_flat.data_ptr()),
        sparse_indices.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        lse.data_ptr<float>(),
        static_cast<float>(sm_scale),
        static_cast<int>(num_tokens));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "attention_kernel_phase5b launch failed: ", cudaGetErrorString(err));
    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_sparse_attention", &dsa_sparse_attention,
          "DSA sparse attention (Phase 5b.3 mma.sync QK tensor cores)");
}

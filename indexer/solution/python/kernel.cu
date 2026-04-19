// DSA topk indexer — Phase 2c: 128-thread cooperative compute with 2-way
// per-slot parallelism.
//
// Builds on Phase 2b's TMA K-load. Block size goes from 32 → 128 (4 warps)
// and per-slot work is split across 2 threads doing 32 heads each. Partial
// sums combine via warp shuffle (`__shfl_xor_sync`). This quarters the
// per-thread compute count (16K FMAs → 4K) while keeping total work the
// same, boosting SM occupancy. Native tcgen05 UMMA with TMEM accumulator
// is deferred to a future phase.
//
// Per-CTA layout:
//   64 slots × 2 threads per slot = 128 threads
//   Thread layout:  tid = slot * 2 + head_group, where head_group ∈ {0,1}
//     head_group 0 owns heads [0..32),  head_group 1 owns heads [32..64)
//   Partial sum per thread = Σ_h  w[h] * ReLU( Σ_d q[h,d] * dequant(k[slot,d]) )
//   Final slot score = partial_0 + partial_1 via `__shfl_xor_sync(0xf...f, _, 1)`.
//
// Compute still FP32 FMA throughout — tensor-core path remains in a separate
// track (mma.sync register layouts need dedicated bring-up time).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <cstring>
#include <cstdint>

namespace {

constexpr int kNumHeads         = 64;
constexpr int kHeadDim          = 128;
constexpr int kPageSize         = 64;
constexpr int kHeadDimWithScale = 132;                         // 128 FP8 + 4 scale bytes
constexpr int kFP8BytesPerPage  = kPageSize * kHeadDim;         // 8192
constexpr int kScaleSectionOff  = kFP8BytesPerPage;             // scales start here within a page
constexpr int kBytesPerPage     = kPageSize * kHeadDimWithScale; // 8448

constexpr int kThreadsPerBlock  = 128;
constexpr int kThreadsPerSlot   = 2;
constexpr int kHeadsPerGroup    = kNumHeads / kThreadsPerSlot;  // 32

// ---------------------------------------------------------------------------
// Inline-PTX helpers for the SM100 async-barrier + bulk-copy path.
// ---------------------------------------------------------------------------

// Convert a generic pointer into a 32-bit shared-memory address (what the
// `[%n]` operands in PTX require for SMEM operations).
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

// Signal one arrival on the barrier and tell it to expect tx_count bytes
// from subsequent cp.async.bulk ops. This must be called AFTER mbarrier_init
// and BEFORE (or alongside) the cp.async.bulk issue.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_count) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 : : "r"(smem_addr(bar)), "r"(tx_count));
}

// Spin until the barrier's current phase completes.
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t done;
    do {
        asm volatile("{ .reg .pred p;"
                     "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                     "  selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(smem_addr(bar)), "r"(phase));
    } while (!done);
}

// Issue a bulk async copy from global to shared memory. On Blackwell this
// dispatches through the TMA unit; unlike `cp.async.bulk.tensor.Nd` this
// variant takes raw byte addresses (no tensor-map descriptor required).
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

// ---------------------------------------------------------------------------
// Phase 2c scoring kernel: 128-thread cooperative compute.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreadsPerBlock, 2)
scoring_kernel_phase2c(
    const uint8_t* __restrict__ q_fp8,        // [B, 64, 128]
    const uint8_t* __restrict__ k_cache,      // [num_pages, 64, 1, 132]
    const float*   __restrict__ weights,      // [B, 64]
    const int32_t* __restrict__ seq_lens,     // [B]
    const int32_t* __restrict__ block_table,  // [B, max_num_pages]
    float*         __restrict__ logits,       // [B, max_seq_len_kv] caller preinits to -inf
    int max_num_pages
) {
    const int page_local = blockIdx.x;
    const int batch      = blockIdx.y;
    const int tid        = threadIdx.x;

    const int seq_len           = seq_lens[batch];
    const int num_pages_for_seq = (seq_len + kPageSize - 1) / kPageSize;
    if (page_local >= num_pages_for_seq) return;

    const int page_id = block_table[batch * max_num_pages + page_local];
    const uint8_t* page_base =
        k_cache + static_cast<int64_t>(page_id) * kBytesPerPage;

    __shared__ float                q_smem[kNumHeads][kHeadDim];        // 32 KiB
    __shared__ alignas(16) uint8_t  k_smem[kBytesPerPage];              // 8448 B
    __shared__ float                w_smem[kNumHeads];                   // 256 B
    __shared__ alignas(8) uint64_t  k_bar;

    // Thread 0: init barrier and kick off the K page TMA bulk load. Because
    // only 1 arrival is expected (thread 0 contributes the expect_tx that
    // the cp.async.bulk completion decrements), count is 1.
    if (tid == 0) {
        mbarrier_init(&k_bar, 1);
        // fence.proxy.async ensures prior SMEM writes (the mbarrier init)
        // are visible before the async proxy (TMA) touches the same SMEM.
        asm volatile("fence.proxy.async.shared::cta;");
        mbarrier_arrive_expect_tx(&k_bar, kBytesPerPage);
        cp_async_bulk_g2s(k_smem, page_base, kBytesPerPage, &k_bar);
    }
    __syncthreads();  // ensure k_bar and TMA issue are visible to all threads

    // Overlap the TMA of K with the cooperative Q FP8→FP32 conversion.
    // With 128 threads, each does 64 Q elements and writes 0.5 weights on avg.
    const int64_t q_row_off =
        static_cast<int64_t>(batch) * kNumHeads * kHeadDim;
    for (int i = tid; i < kNumHeads * kHeadDim; i += kThreadsPerBlock) {
        const int h = i / kHeadDim;
        const int d = i % kHeadDim;
        __nv_fp8_e4m3 v;
        uint8_t byte = q_fp8[q_row_off + h * kHeadDim + d];
        std::memcpy(&v, &byte, 1);
        q_smem[h][d] = static_cast<float>(v);
    }
    for (int i = tid; i < kNumHeads; i += kThreadsPerBlock) {
        w_smem[i] = weights[batch * kNumHeads + i];
    }

    // Wait for K page to be fully delivered before reading from k_smem.
    mbarrier_wait(&k_bar, 0);
    __syncthreads();

    const int     max_seq_len_kv = max_num_pages * kPageSize;
    const int64_t row_base       = static_cast<int64_t>(batch) * max_seq_len_kv;
    const int     page_offset    = page_local * kPageSize;

    // Thread layout: tid = slot * 2 + head_group.
    //   head_group 0 covers heads [0, 32),  head_group 1 covers heads [32, 64).
    const int slot       = tid >> 1;       // 0..63
    const int head_group = tid & 1;        // 0 or 1
    const int h_start    = head_group * kHeadsPerGroup;
    const int h_end      = h_start + kHeadsPerGroup;
    const int tok        = page_offset + slot;

    float partial = 0.0f;
    if (tok < seq_len) {
        float scale;
        std::memcpy(&scale, k_smem + kScaleSectionOff + slot * 4, sizeof(float));
        const uint8_t* kv_ptr = k_smem + slot * kHeadDim;

        for (int h = h_start; h < h_end; ++h) {
            float dot = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                __nv_fp8_e4m3 kv_raw;
                uint8_t kv_byte = kv_ptr[d];
                std::memcpy(&kv_raw, &kv_byte, 1);
                const float k_val = static_cast<float>(kv_raw) * scale;
                dot += q_smem[h][d] * k_val;
            }
            partial += w_smem[h] * fmaxf(dot, 0.0f);
        }
    }

    // Combine the 2 head-groups of this slot via warp shuffle.
    // tid and tid^1 are always in the same warp (since they differ only in
    // the LSB), so `__shfl_xor_sync` with mask 1 exchanges their partials.
    partial += __shfl_xor_sync(0xffffffffu, partial, 1);

    // head_group 0 writes the final slot score.
    if (head_group == 0) {
        if (tok < seq_len) {
            logits[row_base + tok] = partial;
        } else {
            logits[row_base + tok] = -INFINITY;
        }
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// Host entry (pybind).
// -----------------------------------------------------------------------------

void scoring_phase2c(
    torch::Tensor q_u8,
    torch::Tensor k_u8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor logits
) {
    TORCH_CHECK(q_u8.is_cuda() && q_u8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(k_u8.is_cuda() && k_u8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32);
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32);

    const int batch_size    = q_u8.size(0);
    const int max_num_pages = block_table.size(1);
    TORCH_CHECK(q_u8.size(1) == kNumHeads);
    TORCH_CHECK(q_u8.size(2) == kHeadDim);
    TORCH_CHECK(k_u8.size(1) == kPageSize);
    TORCH_CHECK(k_u8.size(3) == kHeadDimWithScale);
    TORCH_CHECK(logits.size(1) == max_num_pages * kPageSize);

    const dim3 grid(max_num_pages, batch_size);
    const dim3 block(kThreadsPerBlock);

    scoring_kernel_phase2c<<<grid, block>>>(
        q_u8.data_ptr<uint8_t>(),
        k_u8.data_ptr<uint8_t>(),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        max_num_pages);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "scoring_kernel_phase2c launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scoring_phase2c", &scoring_phase2c,
          "Phase 2c indexer scoring: 128-thread cooperative FP32 FMA (2 threads/slot, shuffle-reduce)");
}

// DSA topk indexer — Phase 2a: naive FP32 FMA scoring kernel.
//
// This is the first sub-stage of the indexer Stage A (scoring). It's
// intentionally unoptimized: single warp per CTA, plain FP32 FMA, no
// TMA, no UMMA, no tensor cores. Target is correctness against the
// PyTorch reference under rtol=1e-2, atol=1e-2.
//
// Per-page layout of k_index_cache_fp8 (132 bytes per slot):
//   bytes [0 .. page_size*head_dim)                 — FP8 data (slot*head_dim + d)
//   bytes [page_size*head_dim .. page_size*132)     — FP32 scales (4 B per slot)
//
// Grid: (max_num_pages, batch_size). Each CTA = one warp = one page of
// one batch. CTAs with page_local >= num_pages_for_batch early-return.
//
// Subsequent phases will add: TMA pipelined Q/K loads (2b), tcgen05 UMMA
// with TMEM accumulator + per-page epilogue (2c). The CUTLASS/CuTe
// arch headers needed for those phases are validated elsewhere
// (scripts/smoke_compile.py) and are not yet included here — Phase 2a
// has no CUTLASS dependency.

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

// Naive scoring kernel.
//
// Each thread handles two slots (tid and tid+32). For each slot:
//   final_scores[b, tok] = sum_h w[b,h] * ReLU( dot(q[b,h], dequant(kv[page,slot])) )
// with dequant(kv)[d] = float(kv[d]) * scale[slot]. The scale is applied
// elementwise inside the dot to match the PyTorch reference ordering
// (`K_all_dequant = fp8_float * scale`, then matmul).
//
// Q and weights are staged in SMEM once per CTA and reused across all 64
// slots in the page.
__global__ void scoring_kernel_phase2a(
    const uint8_t* __restrict__ q_fp8,        // [B, 64, 128]
    const uint8_t* __restrict__ k_cache,      // [num_pages, 64, 1, 132]
    const float*   __restrict__ weights,      // [B, 64]
    const int32_t* __restrict__ seq_lens,     // [B]
    const int32_t* __restrict__ block_table,  // [B, max_num_pages]
    float*         __restrict__ logits,       // [B, max_seq_len_kv] — caller preinits to -inf
    int max_num_pages
) {
    const int page_local = blockIdx.x;
    const int batch      = blockIdx.y;
    const int tid        = threadIdx.x;                 // 0..31, single warp

    const int seq_len           = seq_lens[batch];
    const int num_pages_for_seq = (seq_len + kPageSize - 1) / kPageSize;
    if (page_local >= num_pages_for_seq) return;

    // Resolve global page id and anchor pointer into the paged KV cache.
    const int page_id = block_table[batch * max_num_pages + page_local];
    const uint8_t* page_base =
        k_cache + static_cast<int64_t>(page_id) * kBytesPerPage;

    // SMEM: Q[64,128] in FP32 (32 KiB) + weights[64] (256 B).
    __shared__ float q_smem[kNumHeads][kHeadDim];
    __shared__ float w_smem[kNumHeads];

    // Cooperative load of Q: 8192 elements / 32 threads = 256/thread.
    const int64_t q_row_off =
        static_cast<int64_t>(batch) * kNumHeads * kHeadDim;
    for (int i = tid; i < kNumHeads * kHeadDim; i += 32) {
        const int h = i / kHeadDim;
        const int d = i % kHeadDim;
        __nv_fp8_e4m3 v;
        uint8_t byte = q_fp8[q_row_off + h * kHeadDim + d];
        std::memcpy(&v, &byte, 1);
        q_smem[h][d] = static_cast<float>(v);
    }
    // 32 threads × 2 heads each = 64 heads. Previous bug: `if (tid < kNumHeads)`
    // only wrote w_smem[0..31], leaving [32..63] uninitialized, which produced
    // ±inf/garbage when multiplied with valid dot products in the accumulation.
    for (int i = tid; i < kNumHeads; i += 32) {
        w_smem[i] = weights[batch * kNumHeads + i];
    }
    __syncthreads();

    const int     max_seq_len_kv = max_num_pages * kPageSize;
    const int64_t row_base       = static_cast<int64_t>(batch) * max_seq_len_kv;
    const int     page_offset    = page_local * kPageSize;

    // Each thread handles 2 slots (tid and tid+32). No continue, no unroll —
    // keep the control flow simple so correctness bugs are easy to spot.
    for (int slot_iter = 0; slot_iter < 2; ++slot_iter) {
        const int slot = tid + slot_iter * 32;
        const int tok  = page_offset + slot;

        if (tok < seq_len) {
            // Per-slot FP32 scale (4 bytes at scale section).
            float scale;
            std::memcpy(&scale, page_base + kScaleSectionOff + slot * 4, sizeof(float));

            const uint8_t* kv_ptr = page_base + slot * kHeadDim;

            float acc = 0.0f;
            for (int h = 0; h < kNumHeads; ++h) {
                float dot = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    __nv_fp8_e4m3 kv_raw;
                    uint8_t kv_byte = kv_ptr[d];
                    std::memcpy(&kv_raw, &kv_byte, 1);
                    const float k_val = static_cast<float>(kv_raw) * scale;
                    dot += q_smem[h][d] * k_val;
                }
                acc += w_smem[h] * fmaxf(dot, 0.0f);
            }
            logits[row_base + tok] = acc;
        } else {
            logits[row_base + tok] = -INFINITY;
        }
    }
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// Host entry (pybind).
// -----------------------------------------------------------------------------

void scoring_phase2a(
    torch::Tensor q_u8,          // [B, 64, 128]               uint8 view of fp8_e4m3
    torch::Tensor k_u8,          // [num_pages, 64, 1, 132]    uint8 view of int8
    torch::Tensor weights,       // [B, 64]                    float32
    torch::Tensor seq_lens,      // [B]                        int32
    torch::Tensor block_table,   // [B, max_num_pages]         int32
    torch::Tensor logits         // [B, max_num_pages*64]      float32 (caller preinits to -inf)
) {
    TORCH_CHECK(q_u8.is_cuda() && q_u8.scalar_type() == torch::kUInt8,
                "q_u8 must be uint8 CUDA tensor");
    TORCH_CHECK(k_u8.is_cuda() && k_u8.scalar_type() == torch::kUInt8,
                "k_u8 must be uint8 CUDA tensor");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32);
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32);

    const int batch_size    = q_u8.size(0);
    const int max_num_pages = block_table.size(1);
    TORCH_CHECK(q_u8.size(1) == kNumHeads,         "num_heads must be 64");
    TORCH_CHECK(q_u8.size(2) == kHeadDim,          "head_dim must be 128");
    TORCH_CHECK(k_u8.size(1) == kPageSize,         "page_size must be 64");
    TORCH_CHECK(k_u8.size(3) == kHeadDimWithScale, "head_dim_with_scale must be 132");
    TORCH_CHECK(block_table.size(0) == batch_size);
    TORCH_CHECK(logits.size(0) == batch_size);
    TORCH_CHECK(logits.size(1) == max_num_pages * kPageSize);

    const dim3 grid(max_num_pages, batch_size);
    const dim3 block(32);

    scoring_kernel_phase2a<<<grid, block>>>(
        q_u8.data_ptr<uint8_t>(),
        k_u8.data_ptr<uint8_t>(),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        max_num_pages);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "scoring_kernel_phase2a launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scoring_phase2a", &scoring_phase2a,
          "Phase 2a indexer scoring: naive FP32 FMA, single warp per page");
}

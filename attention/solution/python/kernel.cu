// DSA sparse attention — Phase 5a: single-warp per-token online-softmax.
//
// MLA-in-MQA sparse attention for DeepSeek-V3.2. ckv_cache is used as
// BOTH the key (concatenated with kpe) AND the value. sparse_indices
// encode (page_idx * 64 + slot); -1 is padding.
//
// Phase 5a design — correctness first:
//   - Grid (num_tokens, 1). One CTA handles one token.
//   - Block 32 threads (1 warp). Sequential over slots within warp.
//   - No TMA, no tensor cores, no warp specialization.
//   - Online softmax (FA-style): running (m, l, O) updated per slot.
//   - FP32 throughout for accumulators; bf16 only on I/O.
//   - SMEM: O[16,512] FP32 = 32 KiB (per-head output accumulator).
//
// Phase 5b-d (deferred): TMA gather, warp specialization, sort-and-
// coalesce page lookup, tcgen05 UMMA for QK and AV, pipelined tiles.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cstdint>
#include <cmath>

namespace {

constexpr int kQoHeads    = 16;
constexpr int kHeadDimCkv = 512;
constexpr int kHeadDimKpe = 64;
constexpr int kTopK       = 2048;

__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, off);
    }
    return x;
}

// Single-warp naive attention kernel.
__global__ void attention_kernel_phase5a(
    const __nv_bfloat16* __restrict__ q_nope,     // [T, 16, 512]
    const __nv_bfloat16* __restrict__ q_pe,       // [T, 16, 64]
    const __nv_bfloat16* __restrict__ ckv_cache,  // [P*64, 512] flat
    const __nv_bfloat16* __restrict__ kpe_cache,  // [P*64, 64]  flat
    const int32_t*       __restrict__ sparse_idx, // [T, 2048]
    __nv_bfloat16*       __restrict__ output,     // [T, 16, 512]
    float*               __restrict__ lse,        // [T, 16]
    float sm_scale,
    int num_tokens
) {
    const int t    = blockIdx.x;
    const int lane = threadIdx.x;  // 0..31
    if (t >= num_tokens) return;

    // Per-head output accumulator: [16, 512] FP32 in SMEM (32 KiB).
    __shared__ float O[kQoHeads][kHeadDimCkv];

    // Clear O.
    for (int i = lane; i < kQoHeads * kHeadDimCkv; i += 32) {
        const int h = i / kHeadDimCkv;
        const int d = i % kHeadDimCkv;
        O[h][d] = 0.0f;
    }
    __syncwarp();

    const int64_t qn_base_t = static_cast<int64_t>(t) * kQoHeads * kHeadDimCkv;
    const int64_t qp_base_t = static_cast<int64_t>(t) * kQoHeads * kHeadDimKpe;
    const int64_t sp_base_t = static_cast<int64_t>(t) * kTopK;

    // Each of 16 heads handled sequentially by the warp.
    for (int h = 0; h < kQoHeads; ++h) {
        const int64_t qn_row = qn_base_t + static_cast<int64_t>(h) * kHeadDimCkv;
        const int64_t qp_row = qp_base_t + static_cast<int64_t>(h) * kHeadDimKpe;

        float m = -INFINITY;
        float l = 0.0f;

        // Iterate all 2048 candidate slots sequentially.
        for (int k = 0; k < kTopK; ++k) {
            int32_t idx = sparse_idx[sp_base_t + k];
            if (idx < 0) continue;

            const int64_t kc_row =
                static_cast<int64_t>(idx) * kHeadDimCkv;
            const int64_t kp_row =
                static_cast<int64_t>(idx) * kHeadDimKpe;

            // Cooperative dot: each lane sums its stride over ckv and kpe.
            float partial = 0.0f;
            for (int d = lane; d < kHeadDimCkv; d += 32) {
                const float q  = __bfloat162float(q_nope[qn_row + d]);
                const float kv = __bfloat162float(ckv_cache[kc_row + d]);
                partial += q * kv;
            }
            for (int d = lane; d < kHeadDimKpe; d += 32) {
                const float q  = __bfloat162float(q_pe[qp_row + d]);
                const float kv = __bfloat162float(kpe_cache[kp_row + d]);
                partial += q * kv;
            }
            const float logit = warp_reduce_sum(partial) * sm_scale;

            // Online softmax state update (every lane holds the same values).
            const float m_new = fmaxf(m, logit);
            const float rescale = expf(m - m_new);  // exp(-inf) → 0, safe.
            const float e = expf(logit - m_new);
            l = l * rescale + e;
            m = m_new;

            // O[h][d] = O[h][d] * rescale + e * ckv_cache[idx][d], lane-strided.
            for (int d = lane; d < kHeadDimCkv; d += 32) {
                const float kv = __bfloat162float(ckv_cache[kc_row + d]);
                O[h][d] = O[h][d] * rescale + e * kv;
            }
        }

        // Write O[h] / l to output (bf16). Guard l==0 (all-padding row).
        const float denom = (l > 0.0f) ? l : 1.0f;
        const int64_t out_row = qn_base_t + static_cast<int64_t>(h) * kHeadDimCkv;
        for (int d = lane; d < kHeadDimCkv; d += 32) {
            const float o = O[h][d] / denom;
            output[out_row + d] = __float2bfloat16(o);
        }

        // lse[t, h] = logsumexp / log(2) = m/log(2) + log2(l).
        // Invalid rows (l==0) keep the preinit -inf.
        if (lane == 0) {
            if (l > 0.0f) {
                constexpr float kInvLog2 = 1.4426950408889634f;  // 1 / ln(2)
                lse[static_cast<int64_t>(t) * kQoHeads + h] = m * kInvLog2 + log2f(l);
            } else {
                lse[static_cast<int64_t>(t) * kQoHeads + h] = -INFINITY;
            }
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

    // Treat ckv/kpe as flat [P*64, D]: sparse_indices encode a global linear index.
    auto ckv_flat = ckv_cache.reshape({-1, kHeadDimCkv});
    auto kpe_flat = kpe_cache.reshape({-1, kHeadDimKpe});

    const dim3 grid(static_cast<unsigned>(num_tokens));
    const dim3 block(32);

    attention_kernel_phase5a<<<grid, block>>>(
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
                "attention_kernel_phase5a launch failed: ", cudaGetErrorString(err));
    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_sparse_attention", &dsa_sparse_attention,
          "DSA sparse attention (Phase 5a naive online softmax)");
}

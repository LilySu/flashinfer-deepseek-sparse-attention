// DSA sparse attention — Phase 5b.2: multi-warp + cp.async.bulk TMA K-load.
//
// Extends Phase 5b.1 by replacing the cooperative bf16 byte-copy K-tile
// load with per-slot cp.async.bulk TMA ops (Blackwell TMA unit). 128
// threads each issue one TMA per chunk: 64 for kc, 64 for kp, all in
// parallel. mbarrier with expect_tx gates the compute until all bytes
// land. Phase 5b.1's multi-warp compute kernel is unchanged.
//
// Per-CTA SMEM (~90 KiB — requires dynamic-SMEM opt-in):
//   qn_smem [16, 512]  BF16 = 16 KiB
//   qp_smem [16,  64]  BF16 =  2 KiB
//   kc_tile [64, 512]  BF16 = 64 KiB
//   kp_tile [64,  64]  BF16 =  8 KiB
//   logits  [16,  64]  FP32 =  4 KiB
//   m_state [16]       FP32 = 64 B
//   l_state [16]       FP32 = 64 B
//   O_acc   [16, 512]  FP32 = 32 KiB
//   Total ≈ 126 KiB
//
// Staged improvements (5b.2, 5b.3, 5c, 5d) layer on top of this.

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
constexpr int kBlockKv    = 64;
constexpr int kNumChunks  = kTopK / kBlockKv;  // 32

constexpr int kThreads    = 128;
constexpr int kWarps      = 4;

struct AttnSmem {
    __nv_bfloat16 qn[kQoHeads][kHeadDimCkv];    // 16 KiB
    __nv_bfloat16 qp[kQoHeads][kHeadDimKpe];    //  2 KiB
    __nv_bfloat16 kc[kBlockKv][kHeadDimCkv];    // 64 KiB
    __nv_bfloat16 kp[kBlockKv][kHeadDimKpe];    //  8 KiB
    float logits[kQoHeads][kBlockKv];           //  4 KiB
    float m_state[kQoHeads];
    float l_state[kQoHeads];
    float O_acc[kQoHeads][kHeadDimCkv];         // 32 KiB
    alignas(8) uint64_t k_bar;                   // TMA mbarrier
};

// Bytes transferred per chunk via cp.async.bulk: 64 slots × (kc + kp).
constexpr uint32_t kTmaBytesPerChunk =
    kBlockKv * (kHeadDimCkv * 2 + kHeadDimKpe * 2);  // 64*(1024+128) = 73728

// ---- TMA / mbarrier inline-PTX helpers ----
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

__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, off);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, off));
    }
    return x;
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

    // --- Stage Q (BF16) + init state ---------------------------------
    const int64_t qn_base = static_cast<int64_t>(t) * kQoHeads * kHeadDimCkv;
    for (int i = tid; i < kQoHeads * kHeadDimCkv; i += kThreads) {
        smem.qn[i / kHeadDimCkv][i % kHeadDimCkv] = q_nope[qn_base + i];
    }
    const int64_t qp_base = static_cast<int64_t>(t) * kQoHeads * kHeadDimKpe;
    for (int i = tid; i < kQoHeads * kHeadDimKpe; i += kThreads) {
        smem.qp[i / kHeadDimKpe][i % kHeadDimKpe] = q_pe[qp_base + i];
    }
    for (int i = tid; i < kQoHeads; i += kThreads) {
        smem.m_state[i] = -INFINITY;
        smem.l_state[i] = 0.0f;
    }
    for (int i = tid; i < kQoHeads * kHeadDimCkv; i += kThreads) {
        smem.O_acc[i / kHeadDimCkv][i % kHeadDimCkv] = 0.0f;
    }

    // Initialize the TMA barrier once per CTA.
    if (tid == 0) {
        mbarrier_init(&smem.k_bar, 1);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    const int64_t sp_base = static_cast<int64_t>(t) * kTopK;
    uint32_t bar_phase = 0;

    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
        const int chunk_off = chunk * kBlockKv;

        // --- Issue TMA bulk loads for 64 kc + 64 kp blocks ------------
        // 128 threads × 1 TMA each: tid 0..63 → kc[slot], tid 64..127 → kp[slot].
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&smem.k_bar, kTmaBytesPerChunk);
        }
        __syncthreads();

        if (tid < kBlockKv) {
            const int s = tid;
            int32_t idx = sparse_idx[sp_base + chunk_off + s];
            int safe_idx = (idx >= 0) ? idx : 0;
            cp_async_bulk_g2s(
                &smem.kc[s][0],
                ckv_flat + static_cast<int64_t>(safe_idx) * kHeadDimCkv,
                kHeadDimCkv * sizeof(__nv_bfloat16),   // 1024 B
                &smem.k_bar);
        } else if (tid < 2 * kBlockKv) {
            const int s = tid - kBlockKv;
            int32_t idx = sparse_idx[sp_base + chunk_off + s];
            int safe_idx = (idx >= 0) ? idx : 0;
            cp_async_bulk_g2s(
                &smem.kp[s][0],
                kpe_flat + static_cast<int64_t>(safe_idx) * kHeadDimKpe,
                kHeadDimKpe * sizeof(__nv_bfloat16),   // 128 B
                &smem.k_bar);
        }

        // Wait for all 128 TMAs this chunk to complete.
        mbarrier_wait(&smem.k_bar, bar_phase);
        bar_phase ^= 1;
        __syncthreads();

        // --- Compute logits[16, 64] via cooperative warp-reduce dot -----
        // 4 warps × 16 slot-cols per warp = 64. Each warp handles 16 slots.
        // For each (h, s) pair, 32 lanes cooperate on the 576-dim dot.
        for (int slot_in_warp = 0; slot_in_warp < kBlockKv / kWarps; ++slot_in_warp) {
            const int s = wid * (kBlockKv / kWarps) + slot_in_warp;
            int32_t idx = sparse_idx[sp_base + chunk_off + s];
            if (idx < 0) {
                // Entire column is -inf for all heads.
                if (lane < kQoHeads) {
                    smem.logits[lane][s] = -INFINITY;
                }
                continue;
            }
            for (int h = 0; h < kQoHeads; ++h) {
                // Dot product over 512 (ckv) + 64 (kpe) = 576 dims.
                float partial = 0.0f;
                for (int d = lane; d < kHeadDimCkv; d += 32) {
                    const float q  = __bfloat162float(smem.qn[h][d]);
                    const float kv = __bfloat162float(smem.kc[s][d]);
                    partial += q * kv;
                }
                for (int d = lane; d < kHeadDimKpe; d += 32) {
                    const float q  = __bfloat162float(smem.qp[h][d]);
                    const float kv = __bfloat162float(smem.kp[s][d]);
                    partial += q * kv;
                }
                const float dot = warp_reduce_sum(partial);
                if (lane == 0) {
                    smem.logits[h][s] = dot * sm_scale;
                }
            }
        }
        __syncthreads();

        // --- Per-head softmax state update + O rescale ---------------
        // One thread per head (tid 0..15).
        __shared__ float rescale_smem[kQoHeads];
        if (tid < kQoHeads) {
            const int h = tid;
            const float m_old = smem.m_state[h];
            float chunk_max = -INFINITY;
            for (int s = 0; s < kBlockKv; ++s) {
                chunk_max = fmaxf(chunk_max, smem.logits[h][s]);
            }
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

        // --- O[h, d] = O[h, d] * rescale + Σ_s logits[h, s] * kc[s, d] --
        // 128 threads × 64 elems each = 8192 = 16 heads × 512 dims.
        for (int i = tid; i < kQoHeads * kHeadDimCkv; i += kThreads) {
            const int h = i / kHeadDimCkv;
            const int d = i % kHeadDimCkv;
            const float rs = rescale_smem[h];
            float acc = smem.O_acc[h][d] * rs;
            for (int s = 0; s < kBlockKv; ++s) {
                const float e = smem.logits[h][s];
                const float kv = __bfloat162float(smem.kc[s][d]);
                acc += e * kv;
            }
            smem.O_acc[h][d] = acc;
        }
        __syncthreads();
    }

    // --- Final normalize + write output + lse -------------------------
    for (int i = tid; i < kQoHeads * kHeadDimCkv; i += kThreads) {
        const int h = i / kHeadDimCkv;
        const int d = i % kHeadDimCkv;
        const float denom = (smem.l_state[h] > 0.0f) ? smem.l_state[h] : 1.0f;
        const float o = smem.O_acc[h][d] / denom;
        output[qn_base + i] = __float2bfloat16(o);
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
    // Opt in to dynamic SMEM > 48 KiB.
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
          "DSA sparse attention (Phase 5b.1 multi-warp cooperative)");
}

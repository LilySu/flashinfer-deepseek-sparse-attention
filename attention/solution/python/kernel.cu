// DSA sparse attention — sm_100a (Blackwell) kernel.
//
// MLA-in-MQA sparse attention for DeepSeek-V3.2. ckv_cache is used as both
// the key component (concatenated with kpe) AND the value — load each ckv
// tile once per iteration and reuse it for QK and AV passes.
//
// CUTLASS/CuTe arch-layer headers are used as PTX-wrapper building blocks
// per contest rule (yongwww, 2026-04-15). CUTLASS is BSD-3-Clause; original
// copyright notices are retained in the upstream headers.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

// Pure PTX wrappers — safe across sm_90 / sm_100
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

// Blackwell-native arch headers (sm_100a)
#include <cute/arch/cluster_sm100.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

// ---------------------------------------------------------------------------
// Kernel stub — skeleton only.
// Returns zeroed output + -inf LSE. Correctness logic to be added in
// subsequent commits (three-role warp specialization, online softmax,
// sort-and-coalesce TMA gather, bf16 UMMA with TMEM accumulator).
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> dsa_sparse_attention(
    torch::Tensor q_nope,         // [T, 16, 512] bf16
    torch::Tensor q_pe,           // [T, 16, 64]  bf16
    torch::Tensor ckv_cache,      // [P, 64, 512] bf16
    torch::Tensor kpe_cache,      // [P, 64, 64]  bf16
    torch::Tensor sparse_indices, // [T, 2048]    int32
    double sm_scale
) {
    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be on CUDA");
    TORCH_CHECK(q_nope.size(1) == 16, "num_qo_heads must be 16");
    TORCH_CHECK(q_nope.size(2) == 512, "head_dim_ckv must be 512");
    TORCH_CHECK(q_pe.size(2) == 64, "head_dim_kpe must be 64");

    const int64_t num_tokens = q_nope.size(0);

    auto out_opts = torch::TensorOptions()
                        .dtype(torch::kBFloat16)
                        .device(q_nope.device());
    auto lse_opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(q_nope.device());

    auto output = torch::zeros({num_tokens, 16, 512}, out_opts);
    auto lse = torch::full({num_tokens, 16}, -INFINITY, lse_opts);

    // TODO: fused sparse attention with MLA-in-MQA trick (ckv as K and V).
    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_sparse_attention", &dsa_sparse_attention,
          "DSA sparse attention for DeepSeek-V3.2 (sm_100a)");
}

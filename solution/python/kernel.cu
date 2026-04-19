// DSA topk indexer — sm_100a (Blackwell) kernel.
//
// This file uses CUTLASS/CuTe arch-layer headers as PTX-wrapper building
// blocks (cluster sync, TMA descriptors, MMA opcode encoding). Per the
// contest rule (yongwww, 2026-04-15), header-only use of CUTLASS/CuTe as
// building blocks is permitted; the kernel logic below is our own.
//
// CUTLASS is BSD-3-Clause licensed. When headers are included here, their
// original copyright/license notices are retained in the upstream files.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

// ---- Pure PTX wrappers — safe across sm_90 / sm_100 ------------------------
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

// ---- Blackwell-native arch headers (sm_100a) -------------------------------
// If these fail to resolve in the eval env, vendor CUTLASS into
// solution/python/third_party/cutlass/include and add it to
// extra_include_paths in binding.py.
#include <cute/arch/cluster_sm100.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

// ---------------------------------------------------------------------------
// Kernel stub — skeleton only.
// Returns a [B, 2048] int32 tensor filled with -1 (padding sentinel).
// Correctness logic to be added in subsequent commits.
// ---------------------------------------------------------------------------

torch::Tensor dsa_topk_indexer(
    torch::Tensor q_index_fp8,        // [B, 64, 128]               float8_e4m3fn
    torch::Tensor k_index_cache_fp8,  // [num_pages, 64, 1, 132]    uint8
    torch::Tensor weights,            // [B, 64]                    float32
    torch::Tensor seq_lens,           // [B]                        int32
    torch::Tensor block_table         // [B, max_num_pages]         int32
) {
    TORCH_CHECK(q_index_fp8.is_cuda(), "q_index_fp8 must be on CUDA");
    TORCH_CHECK(q_index_fp8.size(1) == 64, "num_index_heads must be 64");
    TORCH_CHECK(q_index_fp8.size(2) == 128, "index_head_dim must be 128");

    const int64_t batch_size = q_index_fp8.size(0);
    constexpr int64_t topk = 2048;

    auto opts = torch::TensorOptions()
                    .dtype(torch::kInt32)
                    .device(q_index_fp8.device());
    auto topk_indices = torch::full({batch_size, topk}, -1, opts);

    // TODO: fused dequant + FP8 MMA (tcgen05) + weighted-sum + topk kernel.
    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_topk_indexer", &dsa_topk_indexer,
          "DSA top-K indexer for DeepSeek-V3.2 (sm_100a)");
}

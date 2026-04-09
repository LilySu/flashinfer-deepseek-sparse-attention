"""DSA TopK Indexer — PyTorch + CUDA dequant.

Step 1 optimization: CUDA FP8 dequant kernel (bitwise exact, 5-6x faster).
GEMM + epilogue + topk stay in PyTorch for guaranteed correctness.
"""

import os
import torch
import tvm_ffi
import tvm_ffi.cpp

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64
TOPK = 2048

# ─── Compile and load CUDA dequant kernel ───
_dequant_module = None

def _get_dequant():
    global _dequant_module
    if _dequant_module is not None:
        return _dequant_module

    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(cuda_dir, "kernel.cu")

    lib_path = tvm_ffi.cpp.build(
        "dsa_dequant",
        cuda_files=[kernel_path],
        extra_cuda_cflags=["-O2", "-gencode=arch=compute_100,code=sm_100"],
    )
    _dequant_module = tvm_ffi.load_module(str(lib_path))
    return _dequant_module


def _dequant_fp8_cuda(k_index_cache_fp8):
    """FP8 dequant via CUDA kernel. Returns [num_pages, 64, 128] float32."""
    mod = _get_dequant()
    pages_flat = k_index_cache_fp8.view(torch.uint8).reshape(-1)
    num_pages = pages_flat.numel() // (PAGE_SIZE * (HEAD_DIM + 4))
    output = torch.empty(num_pages, PAGE_SIZE, HEAD_DIM,
                         device=pages_flat.device, dtype=torch.float32)
    mod.dequant_fp8(pages_flat, output)
    torch.cuda.synchronize()
    return output


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """DSA TopK Indexer — contest entry point."""
    B = q_index_fp8.shape[0]

    # CUDA dequant (bitwise exact, faster than Python)
    K_all = _dequant_fp8_cuda(k_index_cache_fp8)

    # Q FP8 → FP32
    q = q_index_fp8.to(torch.float32)

    # Per-batch pipeline (PyTorch — guaranteed correct)
    topk_indices.fill_(-1)

    for b in range(B):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]
        K = K_paged.reshape(-1, HEAD_DIM)[:seq_len]

        q_b = q[b]
        scores = q_b @ K.T
        scores_relu = torch.relu(scores)

        w = weights[b]
        weighted_scores = scores_relu * w[:, None]
        final_scores = weighted_scores.sum(dim=0)

        actual_topk = min(TOPK, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // PAGE_SIZE
        offset_per_token = topk_idx % PAGE_SIZE
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * PAGE_SIZE + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

"""DSA TopK Indexer — Hybrid CUDA dequant: bulk for large B, per-batch for small B.

For small workloads (B × max_pages < total_pages / 2): dequant per-batch (less work).
For large workloads: dequant all pages upfront (less launch overhead).
"""

import os
import torch
import tvm_ffi
import tvm_ffi.cpp

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64
TOPK = 2048
PAGE_BYTES_RAW = PAGE_SIZE * (HEAD_DIM + 4)

_cuda_module = None

def _get_module():
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module

    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(cuda_dir, "kernel.cu")

    lib_path = tvm_ffi.cpp.build(
        "dsa_kernels",
        cuda_files=[kernel_path],
        extra_cuda_cflags=["-O2", "-gencode=arch=compute_100,code=sm_100"],
    )
    _cuda_module = tvm_ffi.load_module(str(lib_path))
    return _cuda_module


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    B = q_index_fp8.shape[0]
    device = q_index_fp8.device
    mod = _get_module()

    # CUDA: Q FP8→FP32
    q = torch.empty(B, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    mod.convert_q(q_index_fp8.view(torch.uint8), q)

    seq_lens_cpu = seq_lens.cpu().tolist()

    num_pages_total = k_index_cache_fp8.shape[0]
    max_pages_needed = max((int(sl) + PAGE_SIZE - 1) // PAGE_SIZE for sl in seq_lens_cpu)
    total_pages_used = B * max_pages_needed  # upper bound

    # Decide: bulk dequant all vs per-batch dequant
    use_bulk = total_pages_used > num_pages_total // 2

    if use_bulk:
        # Bulk dequant all pages (better for large B)
        pages_flat = k_index_cache_fp8.view(torch.uint8).reshape(-1)
        K_all = torch.empty(num_pages_total, PAGE_SIZE, HEAD_DIM, device=device, dtype=torch.float32)
        mod.dequant_fp8(pages_flat, K_all)
        torch.cuda.synchronize()

    kv_flat_pages = k_index_cache_fp8.view(torch.uint8).reshape(num_pages_total, PAGE_BYTES_RAW)

    topk_indices.fill_(-1)

    for b in range(B):
        seq_len = seq_lens_cpu[b]
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        page_indices = block_table[b, :num_pages_for_seq].long()

        if use_bulk:
            K_paged = K_all[page_indices]
        else:
            # Per-batch dequant (better for small B)
            selected_pages = kv_flat_pages[page_indices].reshape(-1)
            n_pg = page_indices.shape[0]
            K_paged = torch.empty(n_pg, PAGE_SIZE, HEAD_DIM, device=device, dtype=torch.float32)
            mod.dequant_fp8(selected_pages, K_paged)

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

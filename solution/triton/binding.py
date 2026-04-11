"""DSA TopK Indexer — Batch-size dispatch: per-batch for B<=2, batched for B>=3.

Small B (1-2): per-batch loop with fused CUDA kernels (best for tiny workloads).
Large B (3+): fully batched pipeline with torch.bmm (eliminates launch overhead).
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

    q = torch.empty(B, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    mod.convert_q(q_index_fp8.view(torch.uint8), q)

    seq_lens_cpu = seq_lens.cpu().tolist()
    max_pages = max((int(sl) + PAGE_SIZE - 1) // PAGE_SIZE for sl in seq_lens_cpu)

    kv_cache_flat = k_index_cache_fp8.view(torch.uint8).reshape(-1)

    topk_indices.fill_(-1)

    if B <= 2:
        _kernel_perbatch(mod, q, kv_cache_flat, weights, seq_lens_cpu,
                         block_table, topk_indices, B, max_pages, device)
    else:
        _kernel_batched(mod, q, kv_cache_flat, weights, seq_lens,
                        seq_lens_cpu, block_table, topk_indices, B,
                        max_pages, device)


def _kernel_perbatch(mod, q, kv_cache_flat, weights, seq_lens_cpu,
                     block_table, topk_indices, B, max_pages, device):
    """Per-batch path: best for B=1-2 (fused kernels, no batching overhead)."""
    K_paged_buf = torch.empty(max_pages, PAGE_SIZE, HEAD_DIM,
                              device=device, dtype=torch.float32)

    for b in range(B):
        seq_len = seq_lens_cpu[b]
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        page_indices = block_table[b, :num_pages_for_seq].long()

        K_paged = K_paged_buf[:num_pages_for_seq]
        mod.gather_dequant_fp8(kv_cache_flat, page_indices, K_paged)

        K = K_paged.reshape(-1, HEAD_DIM)[:seq_len]
        scores = q[b] @ K.T
        scores.clamp_(min=0)
        scores.mul_(weights[b][:, None])
        final_scores = scores.sum(dim=0)

        actual_topk = min(TOPK, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        mod.fused_index_remap(topk_idx, page_indices,
                              topk_indices[b, :actual_topk])


def _kernel_batched(mod, q, kv_cache_flat, weights, seq_lens,
                    seq_lens_cpu, block_table, topk_indices, B,
                    max_pages, device):
    """Batched path: best for B>=3 (~10 launches total instead of 9×B)."""
    max_seq = max_pages * PAGE_SIZE
    actual_topk = min(TOPK, max_seq)

    # Batched gather + dequant
    page_indices_all = block_table[:B, :max_pages].long().reshape(-1)
    K_paged_all = torch.empty(B * max_pages, PAGE_SIZE, HEAD_DIM,
                              device=device, dtype=torch.float32)
    mod.gather_dequant_fp8(kv_cache_flat, page_indices_all, K_paged_all)
    K_all = K_paged_all.reshape(B, max_seq, HEAD_DIM)

    # Batched GEMM
    scores = torch.bmm(q, K_all.transpose(1, 2))

    # Batched ReLU + weight + sum
    scores.clamp_(min=0)
    scores.mul_(weights[:B, :, None])
    final_scores = scores.sum(dim=1)

    # Mask positions beyond seq_len
    positions = torch.arange(max_seq, device=device).unsqueeze(0)
    final_scores[positions >= seq_lens[:B].unsqueeze(1)] = float('-inf')

    # Batched topk
    topk_vals, topk_idx = torch.topk(final_scores, actual_topk, dim=1)

    # Vectorized index remap
    page_local = topk_idx // PAGE_SIZE
    offset = topk_idx % PAGE_SIZE
    bt = block_table[:B, :max_pages].long()
    global_pages = torch.gather(bt, 1, page_local)
    topk_tokens = (global_pages * PAGE_SIZE + offset).to(torch.int32)
    topk_tokens[topk_vals == float('-inf')] = -1
    topk_indices[:B, :actual_topk] = topk_tokens

"""DSA topk indexer — pure-Python reference implementation.

This is the baseline submission used to validate end-to-end eval plumbing
(pack → Modal → benchmark → correctness check). Algorithmic content matches
the reference in the contest definition verbatim; expected speedup ~1.0x.
CUDA/CUTLASS-based fused kernels will replace this path in subsequent
commits while keeping the same `kernel()` signature.
"""

import torch


def _dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm format.

    Input:  [num_pages, page_size, 1, 132] int8 (interpret bytes as uint8)
            Memory layout per page: [fp8_data (P*128 bytes), scales (P*4 bytes)]
    Output: [num_pages, page_size, 128] float32
    """
    k_u8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_u8.shape
    head_dim = head_dim_sf - 4  # 128

    kv_flat = k_u8.view(num_pages, page_size * head_dim_sf)

    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = (
        fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    )
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = (
        scale_bytes.view(num_pages, page_size, 4).view(torch.float32)
    )  # [num_pages, page_size, 1]

    return fp8_float * scale


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    """DSA topk indexer entry point (reference implementation).

    Returns:
        (topk_indices,)
        topk_indices: [batch_size, 2048] int32; -1 marks padding when
                      seq_lens[b] < 2048.
    """
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32)                  # [B, H, D]
    k_all = _dequant_fp8_kv_cache(k_index_cache_fp8)   # [num_pages, P, D]

    topk_indices = torch.full(
        (batch_size, topk), -1, dtype=torch.int32, device=device
    )

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        k_paged = k_all[page_indices]                   # [P_b, P, D]
        k = k_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, D]

        q_b = q[b]                                      # [H, D]
        scores = q_b @ k.T                              # [H, seq_len]
        scores_relu = torch.relu(scores)
        w = weights[b]                                  # [H]
        weighted = scores_relu * w[:, None]
        final_scores = weighted.sum(dim=0)              # [seq_len]

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_tok = topk_idx // page_size
        offset_per_tok = topk_idx % page_size
        global_page = page_indices[page_idx_per_tok]
        topk_tokens = global_page * page_size + offset_per_tok

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)

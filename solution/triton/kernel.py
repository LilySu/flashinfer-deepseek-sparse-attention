"""DSA TopK Indexer - corrected PyTorch baseline, then Triton optimization."""

import torch

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64
TOPK = 2048


def _dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm packed format."""
    k_u8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_u8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_u8.view(num_pages, page_size * head_dim_sf)

    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    B = q_index_fp8.shape[0]
    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32)
    K_all = _dequant_fp8_kv_cache(k_index_cache_fp8)

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

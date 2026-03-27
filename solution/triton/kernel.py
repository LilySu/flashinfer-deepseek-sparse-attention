"""
DSA TopK Indexer — Day-1 Baseline (Batched PyTorch)

Implements dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.
DPS entry point: kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)

Key formula per batch element b:
  scores[h, t] = q[b, h, :] dot K[t, :]   (no scale — baked into weights)
  final[t] = sum_h( ReLU(scores[h, t]) * weights[b, h] )
  topk_indices[b, :] = top-2048 token indices (global: page_id * page_size + offset)
"""

import torch


PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64
TOPK = 2048


def _dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm packed format.

    Input layout per page (page_size * 132 bytes):
      [0 : page_size * 128]  FP8 E4M3 data  (all tokens, then next token...)
      [page_size * 128 : ]   float32 scales  (one per token)

    The tensor [num_pages, page_size, 1, 132] is stored row-major, so
    viewing as [num_pages, page_size * 132] gives the flat byte layout.
    """
    k_u8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_u8.shape
    head_dim = head_dim_sf - 4  # 128

    kv_flat = k_u8.view(num_pages, page_size * head_dim_sf)

    # FP8 data: first page_size * head_dim bytes
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.float()

    # Scale: last page_size * 4 bytes → float32
    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [num_pages, page_size, 1]

    return fp8_float * scale  # [num_pages, page_size, head_dim]


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """DSA TopK Indexer — DPS entry point.

    Args:
        q_index_fp8:        [batch_size, 64, 128]  float8_e4m3fn
        k_index_cache_fp8:  [num_pages, 64, 1, 132]  int8 (deep_gemm packed FP8)
        weights:            [batch_size, 64]  float32
        seq_lens:           [batch_size]  int32
        block_table:        [batch_size, max_num_pages]  int32
        topk_indices:       [batch_size, 2048]  int32  (OUTPUT — write in-place)
    """
    B = q_index_fp8.shape[0]
    num_pages_total = k_index_cache_fp8.shape[0]
    max_num_pages = block_table.shape[1]
    max_seq_len = max_num_pages * PAGE_SIZE
    device = q_index_fp8.device

    # --- Dequantize ---
    q = q_index_fp8.float()  # [B, 64, 128]
    K_all = _dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages_total, 64, 128]

    # --- Gather K pages for all batch elements ---
    page_ids = block_table.long().clamp(0, num_pages_total - 1)  # [B, max_num_pages]
    K_gathered = K_all[page_ids]  # [B, max_num_pages, 64, 128]
    K_flat = K_gathered.view(B, max_seq_len, HEAD_DIM)  # [B, max_seq_len, 128]

    # --- Batched GEMM: q @ K.T ---
    scores = torch.bmm(q, K_flat.transpose(1, 2))  # [B, 64, max_seq_len]

    # --- ReLU + weighted sum across heads ---
    scores = torch.relu(scores)  # [B, 64, max_seq_len]
    final_scores = torch.einsum('bht,bh->bt', scores, weights)  # [B, max_seq_len]

    # --- Mask positions beyond each sequence's length ---
    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)  # [1, max_seq_len]
    seq_lens_long = seq_lens.long()
    mask = positions >= seq_lens_long.unsqueeze(1)  # [B, max_seq_len]
    final_scores.masked_fill_(mask, float('-inf'))

    # --- TopK ---
    actual_k = min(TOPK, max_seq_len)
    _, topk_local = torch.topk(final_scores, k=actual_k, dim=1)  # [B, actual_k]

    # --- Convert local indices to global token indices ---
    # local index i maps to: page = i // PAGE_SIZE, offset = i % PAGE_SIZE
    # global token = physical_page_id * PAGE_SIZE + offset
    page_local = topk_local // PAGE_SIZE  # [B, actual_k] — which page (0..max_num_pages-1)
    offset = topk_local % PAGE_SIZE       # [B, actual_k] — token within page

    # Look up physical page IDs from block_table
    global_page_id = torch.gather(page_ids, 1, page_local)  # [B, actual_k]
    result = (global_page_id * PAGE_SIZE + offset).to(torch.int32)  # [B, actual_k]

    # --- Write output ---
    topk_indices.fill_(-1)
    topk_indices[:, :actual_k] = result

    # Mark padding positions as -1 for sequences shorter than TOPK
    actual_topk_per_batch = torch.clamp(seq_lens_long, max=TOPK)  # [B]
    idx_range = torch.arange(TOPK, device=device).unsqueeze(0)    # [1, 2048]
    invalid_mask = idx_range >= actual_topk_per_batch.unsqueeze(1) # [B, 2048]
    topk_indices[invalid_mask] = -1

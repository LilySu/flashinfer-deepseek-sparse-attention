"""DSA TopK Indexer - Triton + PyTorch hybrid."""

import torch
import triton
import triton.language as tl

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64
TOPK = 2048
PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4)  # 8448


@triton.jit
def _dsa_score_kernel(
    Q_ptr,
    K_cache_ptr,
    weights_ptr,
    seq_lens_ptr,
    block_table_ptr,
    scores_out_ptr,
    num_pages_total,
    stride_q_b, stride_q_h, stride_q_d,
    stride_w_b, stride_w_h,
    stride_bt_b, stride_bt_p,
    stride_so_b, stride_so_t,
    max_seq_len,
    BLOCK_T: tl.constexpr,
    N_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    PG_BYTES: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    page_start = pid_p * BLOCK_T
    t_local = tl.arange(0, BLOCK_T)
    global_t = page_start + t_local
    d_off = tl.arange(0, D_HEAD)

    page_active = page_start < seq_len

    page_id = tl.load(block_table_ptr + pid_b * stride_bt_b + pid_p * stride_bt_p)
    page_base = page_id * PG_BYTES

    # Load FP8 K data: token t, dim d at byte (page_base + t*128 + d)
    k_ptrs = K_cache_ptr + page_base + t_local[:, None] * D_HEAD + d_off[None, :]
    k_int8 = tl.load(k_ptrs, mask=page_active, other=0)
    k_fp8 = k_int8.to(tl.float8e4nv, bitcast=True)
    k_float = k_fp8.to(tl.float32)

    # Load scale: 4 bytes per token at (page_base + 64*128 + t*4)
    sc_base = K_cache_ptr + page_base + BLOCK_T * D_HEAD
    # Load each scale byte as int8, widen to int32, then mask to unsigned value.
    # Cannot use .to(tl.uint8) — Triton may saturate negative int8 to 0 instead of wrapping.
    # The & 0xFF ensures correct unsigned interpretation: int8(-61) → int32(-61) → & 0xFF → 195.
    sb0 = tl.load(sc_base + t_local * 4 + 0, mask=page_active, other=0).to(tl.int32) & 0xFF
    sb1 = tl.load(sc_base + t_local * 4 + 1, mask=page_active, other=0).to(tl.int32) & 0xFF
    sb2 = tl.load(sc_base + t_local * 4 + 2, mask=page_active, other=0).to(tl.int32) & 0xFF
    sb3 = tl.load(sc_base + t_local * 4 + 3, mask=page_active, other=0).to(tl.int32) & 0xFF
    scale_int = sb0 | (sb1 << 8) | (sb2 << 16) | (sb3 << 24)
    scale = scale_int.to(tl.float32, bitcast=True)

    k_dequant = k_float * scale[:, None]

    # Accumulate scores across heads
    final_scores = tl.zeros([BLOCK_T], dtype=tl.float32)
    for h in range(N_HEADS):
        q_ptrs = Q_ptr + pid_b * stride_q_b + h * stride_q_h + d_off * stride_q_d
        q_fp8 = tl.load(q_ptrs)
        q_float = q_fp8.to(tl.float32)
        dot = tl.sum(k_dequant * q_float[None, :], axis=1)
        dot = tl.maximum(dot, 0.0)
        w_h = tl.load(weights_ptr + pid_b * stride_w_b + h * stride_w_h)
        final_scores += dot * w_h

    # Mask invalid tokens
    mask_valid = global_t < seq_len
    final_scores = tl.where(mask_valid, final_scores, float("-inf"))

    # Write output
    out_ptrs = scores_out_ptr + pid_b * stride_so_b + global_t * stride_so_t
    mask_bounds = global_t < max_seq_len
    tl.store(out_ptrs, final_scores, mask=mask_bounds)


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    B = q_index_fp8.shape[0]
    num_pages_total = k_index_cache_fp8.shape[0]
    max_num_pages = block_table.shape[1]
    max_seq_len = max_num_pages * PAGE_SIZE
    device = q_index_fp8.device

    scores = torch.full((B, max_seq_len), float("-inf"), dtype=torch.float32, device=device)

    grid = (B, max_num_pages)
    _dsa_score_kernel[grid](
        q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, scores,
        num_pages_total,
        q_index_fp8.stride(0), q_index_fp8.stride(1), q_index_fp8.stride(2),
        weights.stride(0), weights.stride(1),
        block_table.stride(0), block_table.stride(1),
        scores.stride(0), scores.stride(1),
        max_seq_len,
        BLOCK_T=PAGE_SIZE,
        N_HEADS=NUM_HEADS,
        D_HEAD=HEAD_DIM,
        PG_BYTES=PAGE_BYTES,
        num_warps=4,
        num_stages=1,
    )

    # Top-K
    actual_k = min(TOPK, max_seq_len)
    _, topk_local = torch.topk(scores, k=actual_k, dim=1)

    # Convert local indices to global token indices
    page_local = topk_local // PAGE_SIZE
    offset = topk_local % PAGE_SIZE
    page_ids = block_table.long()
    global_page_id = torch.gather(page_ids, 1, page_local.long())
    result = (global_page_id * PAGE_SIZE + offset).to(torch.int32)

    topk_indices.fill_(-1)
    topk_indices[:, :actual_k] = result

    # Mark padding for short sequences
    seq_lens_long = seq_lens.long()
    actual_topk_per_batch = torch.clamp(seq_lens_long, max=TOPK)
    idx_range = torch.arange(TOPK, device=device).unsqueeze(0)
    invalid_mask = idx_range >= actual_topk_per_batch.unsqueeze(1)
    topk_indices[invalid_mask] = -1

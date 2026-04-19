"""DSA sparse attention — chunk-vectorized PyTorch implementation.

MLA-in-MQA sparse attention for DeepSeek-V3.2. Replaces the per-token
Python loop with a chunked vectorized path that eliminates per-token
device→host syncs and uses batched matmuls.

A fused CUDA kernel targeting sm_100a (tcgen05 MMA + online softmax +
sort-and-coalesce TMA gather) will replace this while keeping the same
kernel() signature.
"""

import math

import torch

# Chunk of tokens processed per iteration. Bounds peak memory for the
# [T_chunk, topk=2048, head_dim_ckv=512] gather buffer (~128 MB/chunk FP32).
_T_CHUNK = 64


@torch.no_grad()
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """DSA sparse attention entry point.

    Inputs:
      q_nope:         [num_tokens, 16, 512]       bfloat16
      q_pe:           [num_tokens, 16, 64]        bfloat16
      ckv_cache:      [num_pages, 64, 512]        bfloat16
      kpe_cache:      [num_pages, 64, 64]         bfloat16
      sparse_indices: [num_tokens, 2048]          int32  (-1 = padding)
      sm_scale:       scalar                      float32

    Returns:
      (output, lse)
      output: [num_tokens, 16, 512] bfloat16
      lse:    [num_tokens, 16]      float32  (2-based log-sum-exp)
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    device = q_nope.device

    # Flatten paged KV to token-level addressing. sparse_indices encode
    # (page_idx * page_size + slot) already, so a single 1-D lookup works.
    kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv),
        dtype=torch.bfloat16, device=device,
    )
    lse = torch.full(
        (num_tokens, num_qo_heads), -float("inf"),
        dtype=torch.float32, device=device,
    )

    log2 = math.log(2.0)

    # Chunk over tokens to bound peak memory. All ops within a chunk are
    # fully vectorized — no per-token Python iteration / .item() syncs.
    for t0 in range(0, num_tokens, _T_CHUNK):
        t1 = min(t0 + _T_CHUNK, num_tokens)
        T = t1 - t0

        indices = sparse_indices[t0:t1]                    # [T, topk] int32
        valid = indices != -1                              # [T, topk] bool
        # Replace -1 with 0 for safe indexing; the mask zeroes their effect later.
        safe_idx = torch.clamp(indices, min=0).to(torch.long)

        # Gather K and KP for all (token, k) positions.
        kc = kc_all[safe_idx]                              # [T, topk, 512]
        kp = kp_all[safe_idx]                              # [T, topk, 64]

        qn = q_nope[t0:t1].to(torch.float32)               # [T, 16, 512]
        qp = q_pe[t0:t1].to(torch.float32)                 # [T, 16, 64]

        # Batched attention logits.
        logits_kc = torch.bmm(qn, kc.transpose(1, 2))      # [T, 16, topk]
        logits_kp = torch.bmm(qp, kp.transpose(1, 2))      # [T, 16, topk]
        logits = (logits_kc + logits_kp) * sm_scale
        # Invalidate padded positions.
        logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))

        lse[t0:t1] = torch.logsumexp(logits, dim=-1) / log2

        attn = torch.softmax(logits, dim=-1)               # may produce NaN
        attn = torch.nan_to_num(attn, nan=0.0)             # rows with all -inf → 0
        out = torch.bmm(attn, kc)                          # [T, 16, 512]
        output[t0:t1] = out.to(torch.bfloat16)

    return (output, lse)

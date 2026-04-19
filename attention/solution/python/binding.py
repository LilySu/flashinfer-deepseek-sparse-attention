"""DSA sparse attention — pure-Python reference implementation.

MLA-in-MQA sparse attention for DeepSeek-V3.2: compressed KV cache (ckv) is
used as BOTH the key (concatenated with kpe) AND the value. Sparse indices
pre-select the top-K KV entries per token.

Baseline submission to validate end-to-end eval plumbing. Expected speedup
~1.0x. A fused CUDA kernel targeting sm_100a (tcgen05 MMA + online softmax
+ sorted-TMA gather) will replace this path while preserving the signature.
"""

import math

import torch


@torch.no_grad()
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """DSA sparse attention entry point (reference implementation).

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

    # Flatten paged KV to token-level addressing: sparse_indices already
    # encode (page_idx * page_size + slot), so a single linear index lookup
    # is sufficient.
    kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv),
        dtype=torch.bfloat16,
        device=device,
    )
    lse = torch.full(
        (num_tokens, num_qo_heads),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )

    log2 = math.log(2.0)

    for t in range(num_tokens):
        indices = sparse_indices[t]
        valid = indices != -1
        valid_idx = indices[valid]

        if valid_idx.numel() == 0:
            output[t].zero_()
            continue

        tok_idx = valid_idx.to(torch.long)

        kc = kc_all[tok_idx]                              # [K, 512]
        kp = kp_all[tok_idx]                              # [K, 64]
        qn = q_nope[t].to(torch.float32)                  # [16, 512]
        qp = q_pe[t].to(torch.float32)                    # [16, 64]

        logits = (qn @ kc.T) + (qp @ kp.T)                # [16, K]
        logits = logits * sm_scale

        lse[t] = torch.logsumexp(logits, dim=-1) / log2

        attn = torch.softmax(logits, dim=-1)              # [16, K]
        out = attn @ kc                                   # [16, 512]
        output[t] = out.to(torch.bfloat16)

    return (output, lse)

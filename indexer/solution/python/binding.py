"""DSA topk indexer — Phase 2a: CUDA scoring + Python topk/remap.

Stage A (scoring) is a CUDA kernel producing a dense [B, max_seq_len_kv]
FP32 logit tensor with -inf past each row's seq_len. Stage B (top-K +
local→global index remap) stays in PyTorch until Phase 3. The overall
kernel() signature is unchanged from the reference.
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_DIAG = os.environ.get("PHASE_DIAG", "0") == "1"

HERE = Path(__file__).resolve().parent

kPageSize = 64
kTopK = 2048

_MODULE = None


def _cutlass_include_dir():
    # Phase 2b+ will need CUTLASS/CuTe. Phase 2a does not, but leaving the
    # include path plumbed keeps the build command stable across phases.
    try:
        import flashinfer
        p = Path(flashinfer.__file__).resolve().parent / "data" / "cutlass" / "include"
        return str(p) if p.exists() else None
    except ImportError:
        return None


def _build():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    extra_inc = []
    cutlass = _cutlass_include_dir()
    if cutlass:
        extra_inc.append(cutlass)
    _MODULE = load(
        name="dsa_topk_indexer_phase2c",
        sources=[str(HERE / "kernel.cu")],
        extra_include_paths=extra_inc,
        extra_cuda_cflags=[
            "-arch=sm_100a",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )
    return _MODULE


@torch.no_grad()
def _ref_logits(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
                max_num_pages):
    """Pure-PyTorch reference computation of Stage A logits (for diagnostic)."""
    B = q_index_fp8.shape[0]
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4

    k_u8 = k_index_cache_fp8.view(torch.uint8)
    kv_flat = k_u8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_fp32 = fp8_tensor.to(torch.float32)
    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)
    k_all = fp8_fp32 * scale

    q = q_index_fp8.to(torch.float32)
    max_seq_len_kv = max_num_pages * page_size
    out = torch.full((B, max_seq_len_kv), float("-inf"),
                      dtype=torch.float32, device=q_index_fp8.device)
    for b in range(B):
        L = int(seq_lens[b].item())
        if L == 0:
            continue
        npages = (L + page_size - 1) // page_size
        pi = block_table[b, :npages].to(torch.long)
        k_paged = k_all[pi]
        k = k_paged.reshape(-1, head_dim)[:L]
        scores = q[b] @ k.T
        weighted = torch.relu(scores) * weights[b][:, None]
        out[b, :L] = weighted.sum(dim=0)
    return out


_DIAG_LOG = "/tmp/phase2a_diag.jsonl"


def _diag_compare_logits(cuda_logits, q, k_cache, w, sl, bt, max_num_pages):
    """Append per-call mismatch stats to a jsonl file; non-raising."""
    import json
    ref = _ref_logits(q, k_cache, w, sl, bt, max_num_pages)
    B = q.shape[0]
    worst = (-1, 0, 0.0, 0.0, 0.0, 0)
    for b in range(B):
        L = int(sl[b].item())
        if L == 0:
            continue
        r = ref[b, :L]
        c = cuda_logits[b, :L]
        diff = (r - c).abs()
        pos = diff.argmax().item()
        d_max = diff[pos].item()
        if d_max > worst[2]:
            worst = (b, L, d_max, r[pos].item(), c[pos].item(), pos)

    sl_list = sl.cpu().tolist()
    entry = {
        "q_shape": list(q.shape),
        "k_shape": list(k_cache.shape),
        "bt_shape": list(bt.shape),
        "max_L": max(sl_list) if sl_list else 0,
        "min_L": min(sl_list) if sl_list else 0,
        "B": B,
        "num_pages": int(k_cache.shape[0]),
        "worst_b": worst[0],
        "worst_L": worst[1],
        "worst_pos": worst[5],
        "worst_ref": worst[3],
        "worst_cuda": worst[4],
        "worst_diff": worst[2],
    }
    with open(_DIAG_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    """DSA topk indexer entry point.

    Returns:
        (topk_indices,)
        topk_indices: [B, 2048] int32; -1 marks padding when seq_len < 2048.
    """
    mod = _build()
    B = q_index_fp8.shape[0]
    max_num_pages = block_table.shape[1]
    max_seq_len_kv = max_num_pages * kPageSize
    device = q_index_fp8.device

    # Pre-init to -inf so any slot the kernel skips stays as padding.
    logits = torch.full(
        (B, max_seq_len_kv), float("-inf"),
        dtype=torch.float32, device=device,
    )

    # FP8 + packed-FP8-cache tensors come in as fp8_e4m3 / int8; reinterpret
    # to uint8 for byte-level access in CUDA.
    q_u8 = q_index_fp8.view(torch.uint8)
    k_u8 = k_index_cache_fp8.view(torch.uint8)

    mod.scoring_phase2c(q_u8, k_u8, weights, seq_lens, block_table, logits)

    if _DIAG:
        torch.cuda.synchronize()
        _diag_compare_logits(
            logits, q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
            max_num_pages,
        )

    # Stage B (PyTorch; to be replaced by CUDA topk in Phase 3):
    # per-batch topk with actual_topk = min(K, seq_len), then local→global remap.
    topk_indices = torch.full((B, kTopK), -1, dtype=torch.int32, device=device)
    for b in range(B):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue
        actual_topk = min(kTopK, seq_len)
        valid = logits[b, :seq_len]
        _, idx = torch.topk(valid, actual_topk)

        bt_b = block_table[b].to(torch.long)
        page_local = (idx // kPageSize).to(torch.long)
        offset = idx % kPageSize
        global_page = bt_b[page_local]
        topk_tokens = (global_page * kPageSize + offset).to(torch.int32)

        topk_indices[b, :actual_topk] = topk_tokens

    return (topk_indices,)

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
_MODULE_2D = None

# Force-enable Phase 2d for testing via env var. In production, use
# _should_use_2d() heuristic below.
# Phase 2d explored but reverted — bounded-merge algorithm has a set-correctness
# bug when pending values exceed frozen threshold_top1984. Fix requires full
# 4096-pad sort per page (~36 cross-warp syncs, ~1.8ms for long-seq rows), which
# is slower than 2c's DRAM round-trip (~64µs). See kernel_2d.cu comment for
# full diagnosis. binding routes to 2c on all shapes.
_FORCE_2D = os.environ.get("USE_PHASE_2D", "0")  # "1" / "0" / "auto"


def _cutlass_include_dir():
    # Phase 2b+ will need CUTLASS/CuTe. Phase 2a does not, but leaving the
    # include path plumbed keeps the build command stable across phases.
    try:
        import flashinfer
        p = Path(flashinfer.__file__).resolve().parent / "data" / "cutlass" / "include"
        return str(p) if p.exists() else None
    except ImportError:
        return None


_COMMON_NVCC_FLAGS = [
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
]


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
        extra_cuda_cflags=_COMMON_NVCC_FLAGS,
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )
    return _MODULE


def _build_2d():
    global _MODULE_2D
    if _MODULE_2D is not None:
        return _MODULE_2D
    extra_inc = []
    cutlass = _cutlass_include_dir()
    if cutlass:
        extra_inc.append(cutlass)
    _MODULE_2D = load(
        name="dsa_topk_indexer_phase2d",
        sources=[str(HERE / "kernel_2d.cu")],
        extra_include_paths=extra_inc,
        extra_cuda_cflags=_COMMON_NVCC_FLAGS,
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )
    return _MODULE_2D


def _should_use_2d(B, max_seq_len_kv, sum_seq_lens):
    """Dispatch heuristic (placeholder — will be tuned after benchmark).
    2d wins when DRAM round-trip is large (large output logits tensor) AND
    SM fill from B CTAs is non-trivial.
    """
    if _FORCE_2D == "1":
        return True
    if _FORCE_2D == "0":
        return False
    # Heuristic: require both B and total work to be non-trivial for 2d.
    return B >= 8 and max_seq_len_kv >= 4096 and sum_seq_lens >= 32768


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
    B = q_index_fp8.shape[0]
    max_num_pages = block_table.shape[1]
    max_seq_len_kv = max_num_pages * kPageSize
    device = q_index_fp8.device

    # --- Dispatch: Phase 2d (fused scoring+topk) for shapes where it wins --
    # Skip the .item() sync entirely unless 2d is in play.
    if _FORCE_2D != "0" and _should_use_2d(
        B, max_seq_len_kv, int(seq_lens.sum().item())
    ):
        mod2d = _build_2d()
        q_u8 = q_index_fp8.view(torch.uint8)
        k_u8 = k_index_cache_fp8.view(torch.uint8)
        topk_indices = torch.full((B, kTopK), -1, dtype=torch.int32, device=device)
        # Telemetry disabled on production path (empty tensor = skip).
        skip_counts = torch.empty(0, dtype=torch.int32, device=device)
        mod2d.fused_phase2d(q_u8, k_u8, weights, seq_lens, block_table,
                            topk_indices, skip_counts)
        return (topk_indices,)

    mod = _build()

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

    # Stage B — vectorized PyTorch topk + remap (Phase 3).
    #
    # Phase 3b (custom CUDA radix-select topk) was evaluated and not shipped:
    # torch.topk internally dispatches to a well-tuned cub-based block topk.
    # Benchmarking Phase 3 vs hand-rolled cub paths showed parity or regress
    # on our workload distribution (many short rows where torch's short-row
    # fast path dominates, few long rows). Custom CUDA radix-select is the
    # right call only if the scoring kernel is itself rewritten to fuse
    # topk inline and avoid materializing the [B, max_seq_len_kv] logits
    # tensor — that's a fusion problem, not a topk problem, and is a
    # candidate for a future "Phase 2d" fused scoring+topk kernel.
    #
    # No per-batch device→host syncs; single torch.topk over
    # [B, max_seq_len_kv] and vectorized gather for the local→global
    # index conversion.
    #
    # torch.topk requires k ≤ dim size, but many workloads have
    # max_seq_len_kv < kTopK (2048). Clamp to the available size and pad
    # the output to kTopK with -1.
    actual_k = min(kTopK, max_seq_len_kv)
    top_vals, local_idx = torch.topk(logits, actual_k, dim=1)  # [B, actual_k]

    page_local_idx = local_idx // kPageSize
    offset = (local_idx % kPageSize).to(torch.int32)

    bt_long = block_table.to(torch.long)
    global_page = torch.gather(bt_long, 1, page_local_idx)
    topk_tokens = (global_page.to(torch.int32) * kPageSize + offset)

    # Positions where top_vals is -inf correspond to out-of-sequence padding
    # → map to -1 sentinel.
    topk_tokens = torch.where(
        torch.isinf(top_vals),
        torch.full_like(topk_tokens, -1),
        topk_tokens,
    )

    if actual_k < kTopK:
        pad = torch.full(
            (B, kTopK - actual_k), -1, dtype=torch.int32, device=device,
        )
        topk_indices = torch.cat([topk_tokens, pad], dim=1)
    else:
        topk_indices = topk_tokens

    return (topk_indices,)

"""DSA sparse attention entry point.

Two paths are wired up, CUDA is the default:
  - Default (cuda): Step 2a CUDA kernel — 4-warp cooperative + 2-stage
    K TMA pipeline (thread-parallel double-buffer) + mma.sync.m16n8k16
    BF16 tensor cores for QK and AV. 23/23 PASSED.
  - Opt-in (python): chunk-vectorized batched torch.bmm reference. Set
    DSA_ATTN_BACKEND=python to activate.

Phase 5c (sort/coalesce) experiments both regressed (in-kernel bitonic
sort added 1344 syncthreads/token; Python-side torch.sort overhead
outweighed the marginal DRAM-locality gain for our workload
distribution). Kept 5b.3 as the shipping CUDA path.
"""

import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent

_BACKEND = os.environ.get("DSA_ATTN_BACKEND", "cuda")  # "python" | "cuda"
_MODULE = None
_BUILD_FAILED = False


def _cutlass_include_dir():
    try:
        import flashinfer
        p = Path(flashinfer.__file__).resolve().parent / "data" / "cutlass" / "include"
        return str(p) if p.exists() else None
    except ImportError:
        return None


def _build():
    global _MODULE, _BUILD_FAILED
    if _MODULE is not None:
        return _MODULE
    if _BUILD_FAILED:
        return None
    extra_inc = []
    cutlass = _cutlass_include_dir()
    if cutlass:
        extra_inc.append(cutlass)
    try:
        _MODULE = load(
            name="dsa_sparse_attention_baseline_2a",
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
    except Exception:
        _BUILD_FAILED = True
        _MODULE = None
    return _MODULE


_T_CHUNK = 64


@torch.no_grad()
def _vectorized_reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    device = q_nope.device

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

    for t0 in range(0, num_tokens, _T_CHUNK):
        t1 = min(t0 + _T_CHUNK, num_tokens)
        indices = sparse_indices[t0:t1]
        valid = indices != -1
        safe_idx = torch.clamp(indices, min=0).to(torch.long)
        kc = kc_all[safe_idx]
        kp = kp_all[safe_idx]
        qn = q_nope[t0:t1].to(torch.float32)
        qp = q_pe[t0:t1].to(torch.float32)
        logits_kc = torch.bmm(qn, kc.transpose(1, 2))
        logits_kp = torch.bmm(qp, kp.transpose(1, 2))
        logits = (logits_kc + logits_kp) * sm_scale
        logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
        lse[t0:t1] = torch.logsumexp(logits, dim=-1) / log2
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.bmm(attn, kc)
        output[t0:t1] = out.to(torch.bfloat16)
    return output, lse


@torch.no_grad()
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """DSA sparse attention entry point.

    Returns:
      (output, lse)
      output: [num_tokens, 16, 512] bfloat16
      lse:    [num_tokens, 16]      float32  (2-based log-sum-exp)
    """
    sm_scale_f = float(sm_scale.item() if torch.is_tensor(sm_scale) else sm_scale)

    if _BACKEND == "cuda":
        mod = _build()
        if mod is not None:
            out, lse_out = mod.dsa_sparse_attention(
                q_nope.contiguous(), q_pe.contiguous(),
                ckv_cache.contiguous(), kpe_cache.contiguous(),
                sparse_indices.contiguous(), sm_scale_f,
            )
            return (out, lse_out)

    # Default: chunk-vectorized PyTorch.
    out, lse_out = _vectorized_reference(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f,
    )
    return (out, lse_out)

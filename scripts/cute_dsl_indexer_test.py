"""Phase 1 — minimal test of the CuTe DSL indexer kernel.

Goal: bring up the kernel.py @cute.jit on Modal B200 with random inputs,
print errors aggressively. We expect this to fail repeatedly until the
API guesses match the installed nvidia-cutlass-dsl version. Each failure
gives us a specific symbol/API to fix.

Usage: uvx modal run scripts/cute_dsl_indexer_test.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("dsa-cute-dsl-indexer-test")

image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu132:20260401-2c675fb",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install("torch", "numpy")
    .add_local_dir(
        str(PROJECT_ROOT / "indexer" / "solution" / "python" / "cute_dsl"),
        remote_path="/root/cute_dsl",
    )
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test():
    import sys
    import traceback
    sys.path.insert(0, "/root")
    import torch

    print("=" * 70)
    print("STEP 1 — import cute_dsl.kernel")
    print("=" * 70)
    try:
        import cute_dsl.kernel as ker
        import cute_dsl.layouts as lay
        print(f"  imported. NUM_HEADS={lay.NUM_HEADS} HEAD_DIM={lay.HEAD_DIM}")
        print(f"  PAGE_SIZE={lay.PAGE_SIZE} BLOCK_Q={lay.BLOCK_Q}")
    except Exception as e:
        print(f"  IMPORT FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"step": "import", "ok": False, "err": str(e)}

    print()
    print("=" * 70)
    print("STEP 2 — create dummy inputs")
    print("=" * 70)
    NUM_HEADS = lay.NUM_HEADS
    HEAD_DIM = lay.HEAD_DIM
    PAGE_SIZE = lay.PAGE_SIZE
    UMMA_M_PADDED = 128
    L = 4  # multi-CTA batched test

    # CuTe wants K-innermost in mA_mkl (M, K, L) — torch's default contiguous
    # is (M, K, L) → strides (K*L, L, 1) which has L innermost. Wrong for TMA.
    # Allocate as (L, M, K) contiguous (strides (M*K, K, 1), K innermost) and
    # permute(1, 2, 0) to view as (M, K, L) with strides (K, 1, M*K). Same memory.
    q_lmk = torch.randn(L, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device="cuda")
    q_lmk = q_lmk.clamp(-3, 3).to(torch.float8_e4m3fn).contiguous()
    q = q_lmk.permute(1, 2, 0)  # CuTe (M=NUM_HEADS, K=HEAD_DIM, L)
    print(f"  q shape={tuple(q.shape)} strides={q.stride()} dtype={q.dtype}")

    k_lnk = torch.randn(L, PAGE_SIZE, HEAD_DIM, dtype=torch.float32, device="cuda")
    k_lnk = k_lnk.clamp(-3, 3).to(torch.float8_e4m3fn).contiguous()
    k = k_lnk.permute(1, 2, 0)  # CuTe (N=PAGE_SIZE, K=HEAD_DIM, L)
    print(f"  k shape={tuple(k.shape)} strides={k.stride()} dtype={k.dtype}")

    # Output: CuTe (M, N, L) with N innermost (strides (N, 1, M*N)).
    s_lmn = torch.zeros(L, UMMA_M_PADDED, PAGE_SIZE, dtype=torch.float32, device="cuda")
    s_out = s_lmn.permute(1, 2, 0)  # CuTe (M=128, N=64, L)
    print(f"  s_out shape={tuple(s_out.shape)} strides={s_out.stride()}")

    print()
    print("=" * 70)
    print("STEP 3 — convert to CuTe tensors via from_dlpack")
    print("=" * 70)
    try:
        from cutlass.cute.runtime import from_dlpack

        # The fp16_gemm_0.py pattern: mark_layout_dynamic + mark_compact_shape_dynamic.
        cQ = from_dlpack(q, assumed_align=16)
        cK = from_dlpack(k, assumed_align=16)
        cS = from_dlpack(s_out, assumed_align=16)

        print(f"  cQ ok, cK ok, cS ok")
    except Exception as e:
        print(f"  from_dlpack FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"step": "from_dlpack", "ok": False, "err": str(e)}

    print()
    print("=" * 70)
    print("STEP 4 — call run() to JIT-compile")
    print("=" * 70)
    try:
        import time
        t0 = time.perf_counter()
        ker.run(cQ, cK, cS)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        print(f"  run() returned. wall={t:.2f}s")
        print(f"  s_out[:4, :4, 0] = {s_out[:4, :4, 0]}")

        # Reference compute per L-slice
        max_diffs = []
        for l in range(L):
            q_fp32 = q[:, :, l].to(torch.float32)
            k_fp32 = k[:, :, l].to(torch.float32)
            s_ref = q_fp32 @ k_fp32.T  # [NUM_HEADS, PAGE_SIZE]
            s_kernel = s_out[:NUM_HEADS, :, l]
            diff = (s_kernel - s_ref).abs()
            mx = diff.max().item()
            mn = diff.mean().item()
            max_diffs.append(mx)
            print(f"  L={l}: ref mag={s_ref.abs().mean():.3f}, kernel mag={s_kernel.abs().mean():.3f}, diff mean={mn:.3f} max={mx:.3f}")
        all_ok = all(d < 1e-3 for d in max_diffs)
        print(f"  ALL L SLICES MATCH: {all_ok}")

        return {
            "step": "run",
            "ok": True,
            "wall_s": t,
            "all_l_match": all_ok,
            "max_diffs": max_diffs,
        }
    except Exception as e:
        print(f"  run() FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "step": "run",
            "ok": False,
            "err": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }


@app.local_entrypoint()
def main():
    import json
    r = test.remote()
    print()
    print("=" * 70)
    print("LOCAL: result")
    print("=" * 70)
    print(json.dumps(r, indent=2, default=str))

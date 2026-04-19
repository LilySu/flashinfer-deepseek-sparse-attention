"""Diagnostic: compare Phase 2a CUDA scoring output to the PyTorch reference
on every contest workload. Reports per-workload: max_abs_err in the logits
tensor (not topk indices — closer to root cause), and the workload shape
(batch_size, max_num_pages, seq_lens summary).

Runs on Modal B200 under the same image as smoke_compile / run_modal_bench.

Usage:
    uvx modal run scripts/diag_phase2a.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-diag-phase2a")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .pip_install("torch", "numpy", "ninja")
    .run_commands(
        "git clone https://github.com/flashinfer-ai/flashinfer-bench.git /opt/flashinfer-bench",
        "cd /opt/flashinfer-bench && pip install -v -e .",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "indexer" / "solution" / "python"),
        remote_path="/root/solution/python",
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_diag() -> dict:
    import torch
    sys.path.insert(0, "/root/solution/python")
    import binding  # Phase 2a

    from flashinfer_bench import TraceSet
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    defn = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
    workloads = trace_set.workloads.get(defn, [])
    print(f"Found {len(workloads)} workloads")
    if workloads:
        wl0 = workloads[0]
        print(f"Workload type: {type(wl0).__name__}")
        print(f"Workload attrs: {[a for a in dir(wl0) if not a.startswith('_')]}")

    # Pure PyTorch reference logits (the pre-topk computation).
    @torch.no_grad()
    def ref_logits(q_fp8, k_cache_fp8, weights, seq_lens, block_table):
        B = q_fp8.shape[0]
        num_pages, page_size, _, head_dim_sf = k_cache_fp8.shape
        head_dim = head_dim_sf - 4

        k_u8 = k_cache_fp8.view(torch.uint8)
        kv_flat = k_u8.view(num_pages, page_size * head_dim_sf)
        fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
        fp8_tensor = (
            fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
        )
        fp8_fp32 = fp8_tensor.to(torch.float32)
        scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
        scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)
        k_all = fp8_fp32 * scale

        q = q_fp8.to(torch.float32)
        max_num_pages = block_table.shape[1]
        max_seq_len_kv = max_num_pages * page_size
        out = torch.full((B, max_seq_len_kv), float("-inf"),
                          dtype=torch.float32, device=q_fp8.device)
        for b in range(B):
            seq_len = int(seq_lens[b].item())
            if seq_len == 0:
                continue
            npages = (seq_len + page_size - 1) // page_size
            pi = block_table[b, :npages].to(torch.long)
            k_paged = k_all[pi]
            k = k_paged.reshape(-1, head_dim)[:seq_len]
            scores = q[b] @ k.T
            weighted = torch.relu(scores) * weights[b][:, None]
            final_scores = weighted.sum(dim=0)
            out[b, :seq_len] = final_scores
        return out

    # CUDA Phase 2a logits (extract just the scoring step — re-run internal call).
    @torch.no_grad()
    def cuda_logits(q_fp8, k_cache_fp8, weights, seq_lens, block_table):
        mod = binding._build()
        B = q_fp8.shape[0]
        max_num_pages = block_table.shape[1]
        max_seq_len_kv = max_num_pages * 64
        out = torch.full((B, max_seq_len_kv), float("-inf"),
                         dtype=torch.float32, device=q_fp8.device)
        q_u8 = q_fp8.view(torch.uint8)
        k_u8 = k_cache_fp8.view(torch.uint8)
        mod.scoring_phase2a(q_u8, k_u8, weights, seq_lens, block_table, out)
        return out

    # Pick the right attribute once; Trace schema may call it inputs / data / kwargs.
    candidates = ["inputs", "data", "kwargs", "input_tensors", "tensors"]
    input_attr = None
    for name in candidates:
        if hasattr(workloads[0], name):
            val = getattr(workloads[0], name)
            if isinstance(val, dict) and "q_index_fp8" in val:
                input_attr = name
                break
    if input_attr is None:
        # Give up and print everything to diagnose.
        wl0 = workloads[0]
        for a in [x for x in dir(wl0) if not x.startswith("_")]:
            try:
                v = getattr(wl0, a)
                print(f"  {a} : {type(v).__name__} = {str(v)[:200]}")
            except Exception as e:
                print(f"  {a} : <err {e}>")
        raise RuntimeError("Could not locate inputs dict on Workload/Trace")
    print(f"Using workload attr: .{input_attr}")

    passes = 0
    fails = []
    for i, wl in enumerate(workloads):
        inputs = getattr(wl, input_attr)
        q_fp8 = inputs["q_index_fp8"].cuda()
        k_cache = inputs["k_index_cache_fp8"].cuda()
        w = inputs["weights"].cuda()
        sl = inputs["seq_lens"].cuda()
        bt = inputs["block_table"].cuda()

        ref = ref_logits(q_fp8, k_cache, w, sl, bt)
        cud = cuda_logits(q_fp8, k_cache, w, sl, bt)

        # Only compare valid positions (< seq_len per row).
        B = q_fp8.shape[0]
        diffs = []
        for b in range(B):
            L = int(sl[b].item())
            if L == 0:
                continue
            r = ref[b, :L]
            c = cud[b, :L]
            if r.numel() == 0:
                continue
            diff = (r - c).abs().max().item()
            diffs.append(diff)

        max_diff = max(diffs) if diffs else 0.0
        # Tolerance on logits (scale free):
        ok = max_diff < 1e-1
        shape_summary = {
            "B": B,
            "max_num_pages": bt.shape[1],
            "seq_lens": sl.cpu().tolist(),
            "num_pages": k_cache.shape[0],
            "max_seq_len": int(sl.max().item()) if sl.numel() else 0,
            "min_seq_len": int(sl.min().item()) if sl.numel() else 0,
        }
        if ok:
            passes += 1
        else:
            fails.append({
                "workload_idx": i,
                "max_abs_err": max_diff,
                **shape_summary,
            })

        if i % 20 == 0:
            print(f"  [{i}/{len(workloads)}] diff={max_diff:.4e} ok={ok}")

    print(f"\n{passes}/{len(workloads)} workloads matched reference logits (tolerance 1e-1)")
    print(f"{len(fails)} failures")
    return {"passes": passes, "total": len(workloads), "fails": fails}


@app.local_entrypoint()
def main():
    result = run_diag.remote()
    print("\n" + "=" * 60)
    print(f"PASS: {result['passes']}/{result['total']}")
    print(f"FAIL: {len(result['fails'])}")
    print("=" * 60)
    if result["fails"]:
        print("\nFailure distribution by (B, max_num_pages, max_seq_len, min_seq_len):")
        buckets = {}
        for f in result["fails"]:
            key = (f["B"], f["max_num_pages"], f["max_seq_len"], f["min_seq_len"])
            buckets.setdefault(key, []).append(f["max_abs_err"])
        for k in sorted(buckets):
            errs = buckets[k]
            print(f"  B={k[0]:3d}  max_num_pages={k[1]:4d}  max_L={k[2]:6d}  min_L={k[3]:6d}   "
                  f"n={len(errs):3d}  median_err={sorted(errs)[len(errs)//2]:.4e}")

        print("\nFirst 10 failing workloads in detail:")
        for f in result["fails"][:10]:
            print(f"  wl={f['workload_idx']:3d}  B={f['B']:3d}  max_pages={f['max_num_pages']:4d}  "
                  f"seq_lens={f['seq_lens'][:3]}...  max_abs_err={f['max_abs_err']:.4e}")

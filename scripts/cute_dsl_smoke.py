"""Phase 0 — CuTe DSL toolchain smoke test on Modal B200.

Verifies that the CuTe DSL (Python) toolchain is usable on the eval-matched
container before we invest time in writing kernels. Probes:

  1. cutlass imports — which sub-packages load
  2. cutlass / sub-module __version__ strings
  3. a trivial @cute.jit vector-add kernel: JIT compile, launch, numerical match
  4. JIT compile latency on cold call vs steady-state launch latency

If any probe fails, print diagnostics and continue to the next probe so we
get maximum information in one Modal session rather than requiring multiple
round-trips.

Usage: uvx modal run scripts/cute_dsl_smoke.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("dsa-cute-dsl-smoke")

# Eval container image per Yongwww 2026-04-24 — pinned, pip-installs cutlass / cute-dsl.
image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu132:20260401-2c675fb",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install("torch", "numpy")
    .run_commands(
        "pip install --upgrade nvidia-cutlass-dsl || true",
    )
)


@app.function(image=image, gpu="B200:1", timeout=600)
def probe():
    import json
    import time
    import traceback

    results = {}

    # ------------------------------------------------------------------
    # Step 1: try importing cutlass and its sub-packages.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1 — imports")
    print("=" * 70)

    imports = {
        "cutlass": None,
        "cutlass.cute": None,
        "cutlass.pipeline": None,
        "cutlass.utils": None,
        "cutlass.utils.blackwell_helpers": None,
        "cutlass.torch": None,
        "cutlass.cute.nvgpu": None,
        "cutlass.cute.nvgpu.cpasync": None,
        "cutlass.cute.nvgpu.tcgen05": None,
        "cutlass.cute.runtime": None,
    }

    for mod_name in list(imports.keys()):
        try:
            mod = __import__(mod_name, fromlist=["*"])
            version = getattr(mod, "__version__", "n/a")
            imports[mod_name] = {"ok": True, "version": version}
            print(f"  OK   {mod_name:<45} version={version}")
        except Exception as e:
            imports[mod_name] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
            print(f"  FAIL {mod_name:<45} {type(e).__name__}: {e}")

    results["imports"] = imports

    core_imports_ok = (
        imports.get("cutlass", {}).get("ok")
        and imports.get("cutlass.cute", {}).get("ok")
    )
    if not core_imports_ok:
        print("\n!! core cutlass.cute import failed — cannot proceed to launch test")
        return results

    # ------------------------------------------------------------------
    # Step 2: enumerate what's available in key namespaces.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 2 — namespace inspection")
    print("=" * 70)

    import cutlass
    import cutlass.cute as cute

    def ns_dump(mod, name, limit=60):
        try:
            members = [m for m in dir(mod) if not m.startswith("_")]
            print(f"  {name} ({len(members)} public members):")
            for m in members[:limit]:
                kind = type(getattr(mod, m, None)).__name__
                print(f"    {m:<35} ({kind})")
            if len(members) > limit:
                print(f"    ... +{len(members) - limit} more")
        except Exception as e:
            print(f"  {name}: dir() failed: {e}")

    ns_dump(cutlass, "cutlass", limit=40)
    ns_dump(cute, "cutlass.cute", limit=40)

    try:
        from cutlass.cute.nvgpu import tcgen05
        ns_dump(tcgen05, "cutlass.cute.nvgpu.tcgen05", limit=60)
    except Exception as e:
        print(f"  tcgen05 import failed: {e}")

    try:
        from cutlass.cute.nvgpu import cpasync
        ns_dump(cpasync, "cutlass.cute.nvgpu.cpasync", limit=40)
    except Exception as e:
        print(f"  cpasync import failed: {e}")

    try:
        import cutlass.pipeline as pipeline
        ns_dump(pipeline, "cutlass.pipeline", limit=40)
    except Exception as e:
        print(f"  pipeline import failed: {e}")

    # ------------------------------------------------------------------
    # Step 3: trivial @cute.jit vector-add kernel.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 3 — trivial @cute.jit vector-add")
    print("=" * 70)

    try:
        import torch
        from cutlass.cute.runtime import from_dlpack

        # Corrected API pattern: from_dlpack is called OUTSIDE @cute.jit.
        # The @cute.jit function receives CuTe tensor args directly.

        @cute.kernel
        def vec_add_kernel(gA, gB, gC):
            tid = cute.arch.thread_idx()[0]
            gC[tid] = gA[tid] + gB[tid]

        @cute.jit
        def vec_add_host(gA, gB, gC):
            vec_add_kernel(gA, gB, gC).launch(
                grid=(1, 1, 1),
                block=(16, 1, 1),
            )

        N = 16
        a = torch.arange(N, dtype=torch.float32, device="cuda")
        b = torch.arange(N, dtype=torch.float32, device="cuda") * 10
        c = torch.zeros(N, dtype=torch.float32, device="cuda")

        # CuTe conversion happens OUTSIDE jit
        cA = from_dlpack(a)
        cB = from_dlpack(b)
        cC = from_dlpack(c)

        # cold JIT
        t0 = time.perf_counter()
        vec_add_host(cA, cB, cC)
        torch.cuda.synchronize()
        t_cold = time.perf_counter() - t0

        # warm launch (re-use converted CuTe tensors)
        c.zero_()
        cC = from_dlpack(c)
        t0 = time.perf_counter()
        vec_add_host(cA, cB, cC)
        torch.cuda.synchronize()
        t_warm = time.perf_counter() - t0

        expected = a + b
        max_err = (c - expected).abs().max().item()
        ok = max_err < 1e-5

        print(f"  cold JIT+launch: {t_cold*1000:.1f} ms")
        print(f"  warm launch:     {t_warm*1e6:.1f} us")
        print(f"  max_err:         {max_err:.2e}")
        print(f"  result:          {'PASS' if ok else 'FAIL'}")
        print(f"  first 8 outputs: {c[:8].tolist()}")
        print(f"  first 8 expect:  {expected[:8].tolist()}")

        results["vector_add"] = {
            "ok": ok,
            "cold_ms": t_cold * 1000,
            "warm_us": t_warm * 1e6,
            "max_err": max_err,
        }

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        print()
        traceback.print_exc()
        results["vector_add"] = {
            "ok": False,
            "err": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }

    # ------------------------------------------------------------------
    # Step 4: probe the tcgen05 MMA atom constructors the plan needs.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 4 — MMA atom constructor probe")
    print("=" * 70)

    try:
        from cutlass.cute.nvgpu import tcgen05 as _tcgen05

        print("  nvidia-cutlass-dsl version:")
        try:
            import importlib.metadata as im
            print(f"    {im.version('nvidia-cutlass-dsl')}")
        except Exception as e:
            print(f"    (could not resolve: {e})")

        print("  All tcgen05 public members containing 'Mma' or 'mma':")
        for m in sorted(dir(_tcgen05)):
            if m.startswith("_"):
                continue
            if "mma" in m.lower():
                print(f"    {m}")

        print("  All tcgen05 public members containing '8' or 'F8' or 'FP8' or 'E4M3' or 'E5M2':")
        for m in sorted(dir(_tcgen05)):
            if m.startswith("_"):
                continue
            if any(tag in m for tag in ("8", "E4M3", "E5M2", "FP8", "F8")):
                print(f"    {m}")

        for atom in ["MmaF16BF16Op", "MmaF8F6F4Op", "MmaTF32Op", "MmaI8Op",
                     "MmaFP8Op", "MmaE4M3Op"]:
            exists = hasattr(_tcgen05, atom)
            print(f"  tcgen05.{atom}: {'present' if exists else 'MISSING'}")

        for e in ["CtaGroup", "OperandSource", "OperandMajorMode", "Field"]:
            exists = hasattr(_tcgen05, e)
            print(f"  tcgen05.{e}: {'present' if exists else 'MISSING'}")
    except Exception as e:
        print(f"  probe failed: {e}")

    try:
        from cutlass.cute.nvgpu import cpasync as _cp
        for cls in ["CopyBulkTensorTileG2SOp", "CopyBulkTensorTileG2SMulticastOp"]:
            exists = hasattr(_cp, cls)
            print(f"  cpasync.{cls}: {'present' if exists else 'MISSING'}")
    except Exception as e:
        print(f"  cpasync probe failed: {e}")

    try:
        import cutlass.pipeline as _p
        for cls in ["PipelineTmaUmma", "PipelineTmaAsync", "PipelineUmmaAsync",
                    "CooperativeGroup", "NamedBarrier"]:
            exists = hasattr(_p, cls)
            print(f"  pipeline.{cls}: {'present' if exists else 'MISSING'}")
    except Exception as e:
        print(f"  pipeline probe failed: {e}")

    try:
        import cutlass.utils as _u
        for cls in ["TmemAllocator"]:
            exists = hasattr(_u, cls)
            print(f"  utils.{cls}: {'present' if exists else 'MISSING'}")
    except Exception as e:
        print(f"  utils probe failed: {e}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps({
        "imports_ok": {k: v.get("ok", False) for k, v in imports.items()},
        "vector_add_ok": results.get("vector_add", {}).get("ok", False),
        "vector_add_cold_ms": results.get("vector_add", {}).get("cold_ms"),
        "vector_add_warm_us": results.get("vector_add", {}).get("warm_us"),
    }, indent=2))

    return results


@app.local_entrypoint()
def main():
    import json
    r = probe.remote()
    print()
    print("=" * 70)
    print("LOCAL: received result")
    print("=" * 70)
    print(json.dumps(r, indent=2, default=str))

"""Compile-only smoke test for the sm_100a kernel scaffold.

Runs on Modal B200, calls binding._build() to JIT-compile kernel.cu with
-arch=sm_100a and the 5 CUTLASS/CuTe headers. Does NOT run benchmarks.

Outcomes:
  - PASS: CUTLASS resolves, sm_100a compiles, extension loads. Safe to
    proceed with kernel logic.
  - FAIL: header ENOENT → vendor CUTLASS into solution/python/third_party/
  - FAIL: ptxas / nvcc error on sm_100a → check CUDA toolkit version in
    image, may need CUDA 12.9+
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-smoke-compile")

# Match the confirmed-working B200 image (CUDA 13.0 devel, has nvcc + ncu).
# Contest eval uses CUDA 13.2; 13.0 is close enough for a compile probe
# and is the proven-working image from prior Modal runs.
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
        str(PROJECT_ROOT / "solution" / "python"),
        remote_path="/root/solution/python",
    )
)


@app.function(image=image, gpu="B200:1", timeout=600)
def compile_and_load() -> dict:
    """Import binding, trigger JIT compile, attempt a dummy call."""
    import subprocess
    import torch

    print("=" * 60)
    print("CUDA toolkit info")
    print("=" * 60)
    subprocess.run(["nvcc", "--version"], check=False)
    print(f"\ntorch.__version__ = {torch.__version__}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"torch.cuda.get_device_capability(0) = {cap}")

    print("\n" + "=" * 60)
    print("Probing CUTLASS header availability")
    print("=" * 60)
    candidate_paths = [
        "/usr/local/cuda/include/cutlass/arch/barrier.h",
        "/usr/include/cutlass/arch/barrier.h",
        "/opt/cutlass/include/cutlass/arch/barrier.h",
    ]
    for p in candidate_paths:
        print(f"  {p}: {'FOUND' if Path(p).exists() else 'missing'}")

    # Try to find CUTLASS via Python package (flashinfer ships it)
    try:
        import flashinfer
        fi_dir = Path(flashinfer.__file__).parent
        print(f"\n  flashinfer dir: {fi_dir}")
        for pattern in ["**/cutlass/include/cutlass/arch/barrier.h",
                        "**/include/cutlass/arch/barrier.h"]:
            for hit in fi_dir.glob(pattern):
                print(f"    hit: {hit}")
                break
    except ImportError:
        print("\n  flashinfer not importable")

    print("\n" + "=" * 60)
    print("Compiling kernel.cu")
    print("=" * 60)
    import sys as _sys
    _sys.path.insert(0, "/root/solution/python")
    try:
        import binding
        mod = binding._build()
        print(f"\nCompile OK. Extension: {mod}")
    except Exception as e:
        print(f"\nCompile FAILED:\n{type(e).__name__}: {e}")
        return {"status": "compile_failed", "error": str(e)}

    print("\n" + "=" * 60)
    print("Smoke-invoking kernel with dummy inputs")
    print("=" * 60)
    device = torch.device("cuda:0")
    B, H, D, P = 2, 64, 128, 64
    num_pages = 4
    max_pages = 2

    q = torch.zeros(B, H, D, dtype=torch.float8_e4m3fn, device=device)
    k = torch.zeros(num_pages, P, 1, 132, dtype=torch.int8, device=device)
    w = torch.zeros(B, H, dtype=torch.float32, device=device)
    s = torch.zeros(B, dtype=torch.int32, device=device)
    bt = torch.zeros(B, max_pages, dtype=torch.int32, device=device)

    try:
        out = binding.kernel(q, k, w, s, bt)
        print(f"Invoke OK. Output shape: {out[0].shape}, dtype: {out[0].dtype}")
        return {
            "status": "pass",
            "output_shape": list(out[0].shape),
            "output_dtype": str(out[0].dtype),
        }
    except Exception as e:
        print(f"Invoke FAILED:\n{type(e).__name__}: {e}")
        return {"status": "invoke_failed", "error": str(e)}


@app.local_entrypoint()
def main():
    # Mount solution/python at /root/solution/python in the container.
    result = compile_and_load.remote()
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)

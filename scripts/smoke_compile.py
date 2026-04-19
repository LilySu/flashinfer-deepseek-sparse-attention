"""Compile-only smoke test for the sm_100a kernel scaffold.

Runs on Modal B200, JIT-compiles kernel.cu directly via
torch.utils.cpp_extension.load() with -arch=sm_100a and the 5 CUTLASS/CuTe
headers. Does NOT run benchmarks or invoke binding.py.

Usage:
    KERNEL=indexer   uvx modal run scripts/smoke_compile.py
    KERNEL=attention uvx modal run scripts/smoke_compile.py

Outcomes:
  - PASS: CUTLASS resolves, sm_100a compiles, extension loads.
  - FAIL: header ENOENT → vendor CUTLASS into third_party/
  - FAIL: ptxas / nvcc error on sm_100a → check CUDA toolkit version
"""

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KERNEL = os.environ.get("KERNEL", "indexer")

import modal

app = modal.App(f"flashinfer-smoke-{KERNEL}")

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
        str(PROJECT_ROOT / KERNEL / "solution" / "python"),
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
    print("Compiling kernel.cu directly via torch.utils.cpp_extension.load")
    print("=" * 60)
    from torch.utils.cpp_extension import load

    import flashinfer
    cutlass_inc = str(Path(flashinfer.__file__).resolve().parent
                       / "data" / "cutlass" / "include")
    print(f"  cutlass_inc = {cutlass_inc}")

    try:
        mod = load(
            name=f"smoke_kernel_sm100a",
            sources=["/root/solution/python/kernel.cu"],
            extra_include_paths=[cutlass_inc],
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
        print(f"\nCompile OK. Extension: {mod}")
        print(f"  exported fns: {[x for x in dir(mod) if not x.startswith('_')]}")
        return {
            "status": "pass",
            "exports": [x for x in dir(mod) if not x.startswith('_')],
        }
    except Exception as e:
        print(f"\nCompile FAILED:\n{type(e).__name__}: {e}")
        return {"status": "compile_failed", "error": str(e)[:2000]}


@app.local_entrypoint()
def main():
    # Mount solution/python at /root/solution/python in the container.
    result = compile_and_load.remote()
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)

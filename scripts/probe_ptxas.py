"""One-shot nvcc -ptxas-options=-v probe on Modal B200.

Compiles attention/solution/python/kernel.cu directly with nvcc (not via
torch cpp_extension), captures ptxas stderr. Used to verify SMEM/register
usage after changes. Not part of the benchmark path.

Usage: uvx modal run scripts/probe_ptxas.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("dsa-probe-ptxas")

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
        str(PROJECT_ROOT / "attention" / "solution" / "python"),
        remote_path="/root/src",
    )
)


@app.function(image=image, gpu="B200:1", timeout=600)
def probe() -> str:
    import subprocess
    import torch  # to locate include dirs

    torch_inc = str(Path(torch.__file__).resolve().parent / "include")
    torch_api = str(Path(torch_inc) / "torch" / "csrc" / "api" / "include")
    py_inc = subprocess.check_output(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
    ).decode().strip()

    import flashinfer
    cutlass_inc = str(Path(flashinfer.__file__).resolve().parent
                       / "data" / "cutlass" / "include")

    cmd = [
        "nvcc",
        "-arch=sm_100a",
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v",
        "-c", "/root/src/kernel.cu",
        "-o", "/tmp/kernel.o",
        "-I", torch_inc,
        "-I", torch_api,
        "-I", py_inc,
        "-I", cutlass_inc,
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DNDEBUG",  # release-build ptxas numbers (what runs in bench)
    ]
    print(" ".join(cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    print("=== STDOUT ===")
    print(out.stdout)
    print("=== STDERR ===")
    print(out.stderr)
    print(f"=== RETURNCODE: {out.returncode} ===")
    return out.stderr


@app.local_entrypoint()
def main():
    print(probe.remote())

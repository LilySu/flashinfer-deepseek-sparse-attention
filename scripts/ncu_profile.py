"""Phase 6: NCU profile — two-pass triage of DSA kernels.

Pass 1 (this script) — lightweight SoL / occupancy / launch stats sweep
on a single representative workload per kernel. Output: per-kernel
timings and occupancy. Pass 2 (deep dive on top offenders) is left as a
follow-up that re-runs with a full section set per the per-kernel list
in memory/reference_ncu_profiling_strategy.md.

Usage:
    KERNEL=indexer   uvx modal run scripts/ncu_profile.py
    KERNEL=attention uvx modal run scripts/ncu_profile.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KERNEL = os.environ.get("KERNEL", "indexer")

import modal

app = modal.App(f"flashinfer-ncu-{KERNEL}")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
ncu_volume = modal.Volume.from_name("ncu-reports", create_if_missing=True)

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
        remote_path="/root/submission/solution/python",
    )
    .add_local_file(
        str(PROJECT_ROOT / KERNEL / "config.toml"),
        remote_path="/root/submission/config.toml",
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=1800,
    volumes={"/data": trace_volume, "/ncu_reports": ncu_volume},
)
def run_ncu(kernel_kind: str = "indexer") -> dict:
    import subprocess
    import torch
    from pathlib import Path as P

    # 1) Warm up the JIT compile by importing binding.
    sys.path.insert(0, "/root/submission/solution/python")
    import binding
    binding._build()  # ensures kernel.cu is compiled
    print("Kernel JIT'd.")

    # 2) Write a tiny runner script that materializes one workload
    # and invokes kernel() — NCU will attach to this subprocess.
    runner_path = P("/tmp/ncu_runner.py")
    runner_path.write_text(f"""
import sys
sys.path.insert(0, "/root/submission/solution/python")
import torch, binding

# Minimal synthetic workload covering the indexer signature.
B = 4
num_pages = 64
max_num_pages = 16
device = "cuda:0"
{'''
q = torch.zeros(B, 64, 128, dtype=torch.float8_e4m3fn, device=device)
k = torch.zeros(num_pages, 64, 1, 132, dtype=torch.int8, device=device)
w = torch.rand(B, 64, dtype=torch.float32, device=device)
sl = torch.full((B,), 1024, dtype=torch.int32, device=device)
bt = torch.randint(0, num_pages, (B, max_num_pages), dtype=torch.int32, device=device)
for _ in range(6):
    out = binding.kernel(q, k, w, sl, bt)
torch.cuda.synchronize()
print("indexer kernel executed")
''' if kernel_kind == 'indexer' else '''
T = 256
num_pages = 1024
qn = torch.randn(T, 16, 512, dtype=torch.bfloat16, device=device)
qp = torch.randn(T, 16, 64,  dtype=torch.bfloat16, device=device)
ckv = torch.randn(num_pages, 64, 512, dtype=torch.bfloat16, device=device)
kpe = torch.randn(num_pages, 64,  64, dtype=torch.bfloat16, device=device)
si  = torch.randint(0, num_pages * 64, (T, 2048), dtype=torch.int32, device=device)
import os
os.environ["DSA_ATTN_BACKEND"] = "cuda"  # profile the CUDA path
for _ in range(6):
    out, lse = binding.kernel(qn, qp, ckv, kpe, si, 1.0/192**0.5)
torch.cuda.synchronize()
print("attention kernel executed")
'''}
""")

    # 3) Run under NCU: Pass-1 triage — LaunchStats + SpeedOfLight + Occupancy only.
    report_path = f"/ncu_reports/{kernel_kind}_phase6_triage.ncu-rep"
    # Filter to our actual kernel (scoring_kernel_phase2c for indexer,
    # attention_kernel_phase5a for attention). Without this, NCU picks up
    # torch's internal vectorized_elementwise_kernel from preinit ops.
    kernel_regex = (
        "scoring_kernel_phase" if kernel_kind == "indexer"
        else "attention_kernel_phase"
    )
    cmd = [
        "/usr/local/cuda/bin/ncu",
        "--set", "basic",
        "--section", "LaunchStats",
        "--section", "Occupancy",
        "--section", "SpeedOfLight",
        "--kernel-name-base", "mangled",
        "--kernel-name", f"regex:{kernel_regex}",
        "--launch-skip", "2",     # skip JIT warmup (after --kernel-name filter)
        "--launch-count", "1",
        "--cache-control", "all",
        "--replay-mode", "kernel",
        "--target-processes", "all",
        "--clock-control", "none",
        "-f",                          # overwrite existing report
        "--export", report_path,
        "python3", "/tmp/ncu_runner.py",
    ]
    print(f"\nRunning NCU: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print("=== NCU stdout ===")
    print(result.stdout[:4000])
    print("=== NCU stderr ===")
    print(result.stderr[:4000])

    # 4) Summary CSV dump from the .ncu-rep.
    csv_path = f"/ncu_reports/{kernel_kind}_phase6_triage.csv"
    subprocess.run(
        ["/usr/local/cuda/bin/ncu", "--import", report_path, "--csv",
         "--page", "details"],
        capture_output=False, text=True,
    )

    ncu_volume.commit()
    return {
        "report": report_path,
        "csv": csv_path,
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main():
    r = run_ncu.remote(KERNEL)
    print("\n=== RESULT ===")
    print(r)
    print("\nTo download report:")
    print(f"  modal volume get ncu-reports {KERNEL}_phase6_triage.ncu-rep")
    print(f"  ncu-ui {KERNEL}_phase6_triage.ncu-rep")
